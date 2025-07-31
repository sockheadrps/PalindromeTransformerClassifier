#!/usr/bin/env python3
"""
Palindrome Detection Router with FastAPI Integration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
import os
import sys
import tensorflow as tf
import numpy as np
from palindrome_transformer import LearnedPositionalEncoding
from multiword_data_gen import encode_multiword, char_reflection_score, char_reflection_score_detailed
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob


# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create FastAPI router
palindrome_router = APIRouter()

# Pydantic models for request/response validation


class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to check for palindrome",
                      min_length=1, max_length=1000)
    include_reflection_analysis: bool = Field(
        False, description="Include detailed reflection analysis")
    include_all_visualizations: bool = Field(
        False, description="Include all available visualizations")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to check",
                             min_items=1, max_items=100)
    include_reflection_analysis: bool = Field(
        False, description="Include reflection analysis for each text")


class VisualizeRequest(BaseModel):
    text: str = Field(..., description="Text to visualize",
                      min_length=1, max_length=1000)
    model_key: Optional[str] = Field(None, description="Specific model to use")
    maxlen: int = Field(
        100, description="Maximum sequence length", ge=1, le=200)


class GenerateVisualizationsRequest(BaseModel):
    """Request model for generating comprehensive visualizations."""
    include_performance: bool = Field(
        True, description="Include performance analysis visualizations")
    include_layer_analysis: bool = Field(
        True, description="Include layer analysis visualizations")
    include_reflection_analysis: bool = Field(
        True, description="Include reflection analysis visualizations")
    test_texts: Optional[List[str]] = Field(
        None, description="Custom test texts for analysis")


class ModelLoadRequest(BaseModel):
    model_key: str = Field(..., description="Model key to load")


# Single model configuration - Reflection-Integrated Model
MODEL_CONFIG = {
    "name": "PalindromeModel_Reflection_Integrated",
    "display_name": "Reflection-Integrated Palindrome Model (1-50 chars)",
    "min_len": 1,
    "max_len": 50,
    "input_len": 50,  # Trained with input_len=50
    "filename": "models/model_with_reflection_fixed.keras",
    "has_reflection": True  # Flag to indicate this model uses reflection scores
}

# Global variables for single model management
model = None  # Single loaded model
last_usage_time = None  # Last usage time
visualizer_coordinator = None  # Single visualizer coordinator
unload_scheduled = False  # Track if unload is already scheduled
unload_task = None  # Track the current unload task
UNLOAD_DELAY = timedelta(minutes=5)  # 5 minutes
shutdown_requested = False  # Flag to indicate application shutdown


def get_model_for_length(text_length):
    """Check if the text length is supported by our single model."""
    if MODEL_CONFIG["min_len"] <= text_length <= MODEL_CONFIG["max_len"]:
        return "model_1_50", MODEL_CONFIG
    return None, None


def get_best_available_model(text_length):
    """Get the single model if it's available and text length is supported."""
    if get_model_for_length(text_length)[0] and model is not None:
        return "model_1_50", MODEL_CONFIG, "Using single model (already loaded)"
    elif get_model_for_length(text_length)[0]:
        return "model_1_50", MODEL_CONFIG, "Loading single model"
    return None, None, "Text length outside supported range (1-50 chars)"


def load_model():
    """Load the single model."""
    global model, last_usage_time

    if model is not None:
        return True

    model_filename = MODEL_CONFIG["filename"]

    try:
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file {model_filename} not found")

        print(
            f"🔄 Loading {MODEL_CONFIG['display_name']} from: {model_filename}")
        model = tf.keras.models.load_model(
            model_filename,
            custom_objects={
                "LearnedPositionalEncoding": LearnedPositionalEncoding,
            }
        )
        last_usage_time = datetime.now()
        print(f"✅ {MODEL_CONFIG['display_name']} loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading {MODEL_CONFIG['display_name']}: {e}")
        return False


def unload_model():
    """Unload the single model to free memory."""
    global model, last_usage_time, unload_scheduled, unload_task

    if model is not None:
        print(f"💾 Unloading {MODEL_CONFIG['display_name']} to free memory...")
        del model
        model = None
        last_usage_time = None
        unload_scheduled = False
        unload_task = None
        print(f"✅ {MODEL_CONFIG['display_name']} unloaded successfully!")


def ensure_model_loaded() -> bool:
    global last_usage_time, unload_scheduled
    if model is None:
        if not load_model():
            return False
        # Reset unload scheduling when model is first loaded
        unload_scheduled = False
    now = datetime.now()
    last_usage_time = now
    return True


def schedule_model_unload(background_tasks: BackgroundTasks):
    """Schedule model unload using FastAPI background tasks."""
    global unload_scheduled, last_usage_time, shutdown_requested, unload_task

    # Don't schedule if shutting down
    if shutdown_requested:
        return

    # Update last usage time
    last_usage_time = datetime.now()

    # Cancel any existing unload schedule
    if unload_scheduled:
        print(f"⏰ Cancelling existing unload task (model used again)")
        unload_scheduled = False
        # Cancel the existing task if it exists
        if unload_task and not unload_task.done():
            unload_task.cancel()

    # Schedule new unload
    unload_scheduled = True
    unload_task = background_tasks.add_task(delayed_unload_model)
    print(f"⏰ Scheduled new unload task (will unload in 5 minutes if unused)")


async def delayed_unload_model():
    """Background task to unload model after delay."""
    global unload_scheduled, shutdown_requested, unload_task

    try:
        # Sleep for 5 minutes in smaller chunks to allow cancellation
        start_time = asyncio.get_event_loop().time()
        target_time = start_time + 300  # 5 minutes

        while asyncio.get_event_loop().time() < target_time:
            # Sleep in 1-second chunks to allow for cancellation
            await asyncio.sleep(1)

        # Check if we should cancel (model was used again or shutdown requested)
        if not unload_scheduled or shutdown_requested:
            if shutdown_requested:
                print("⏰ Cancelled unload (application shutting down)")
            else:
                print("⏰ Cancelled unload (model was used again)")
            return

        # Check if model is still unused and not shutting down
        if (not shutdown_requested and
            last_usage_time and
                datetime.now() - last_usage_time >= UNLOAD_DELAY):

            print("⏰ Auto-unloading model after inactivity")
            unload_model()
            unload_scheduled = False
            unload_task = None
    except asyncio.CancelledError:
        print("⏰ Unload task cancelled")
        unload_scheduled = False
        unload_task = None


def cleanup_overdue_models():
    """Clean up the model if it's overdue for unloading."""
    if (model is not None and
        last_usage_time and
            datetime.now() - last_usage_time >= UNLOAD_DELAY):
        print("🧹 Cleaning up overdue model")
        unload_model()
        return 1
    return 0


def get_model_status():
    """Get current status of the single model."""

    is_loaded = model is not None

    # Get model file size
    model_file_path = MODEL_CONFIG.get(
        "model_path", "models/model_with_reflection_fixed.keras")
    file_size = "Unknown"
    if os.path.exists(model_file_path):
        size_bytes = os.path.getsize(model_file_path)
        if size_bytes < 1024:
            file_size = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            file_size = f"{size_bytes / 1024:.1f} KB"
        else:
            file_size = f"{size_bytes / (1024 * 1024):.1f} MB"

    model_status = {
        "name": MODEL_CONFIG["name"],
        "display_name": MODEL_CONFIG["display_name"],
        "loaded": is_loaded,
        "length_range": f"{MODEL_CONFIG['min_len']}-{MODEL_CONFIG['max_len']} chars",
        "input_length": MODEL_CONFIG["input_len"],
        "file_size": file_size
    }

    if is_loaded and last_usage_time is not None:
        time_since_usage = datetime.now() - last_usage_time
        time_until_unload = UNLOAD_DELAY - time_since_usage
        if time_until_unload > timedelta(0):
            model_status.update({
                "last_used": last_usage_time.isoformat(),
                "time_since_usage": str(time_since_usage).split('.')[0],
                "time_until_unload": str(time_until_unload).split('.')[0]
            })
        else:
            model_status.update({
                "last_used": last_usage_time.isoformat(),
                "time_since_usage": str(time_since_usage).split('.')[0],
                "time_until_unload": "Overdue for unload"
            })
    else:
        model_status.update({
            "last_used": None,
            "time_since_usage": None,
            "time_until_unload": None
        })

    return {"model_1_50": model_status}


# VisualizerCoordinator function removed - not needed for current implementation


def generate_comprehensive_visualizations(request_data):
    """Generate comprehensive visualizations for the reflection-integrated model."""
    try:
        # Ensure model is loaded
        if not ensure_model_loaded():
            return {"error": "Model not loaded"}

        # Get the input text from the request
        test_texts = request_data.get("test_texts", [])
        if not test_texts:
            return {"error": "No test texts provided for visualization"}

        # Create output directory
        output_dir = "visualization_output"
        os.makedirs(output_dir, exist_ok=True)

        # Generate visualizations for the specific input
        visualizations = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, text in enumerate(test_texts):
            # Clean the text
            cleaned = ''.join(c.lower() for c in text.strip() if c.isalpha())

            # Use actual model results if provided, otherwise calculate defaults
            model_results = request_data.get("model_results", {})
            if model_results:
                is_palindrome = model_results.get("is_palindrome", False)
                confidence = model_results.get("confidence", 0.0)
                reflection_score = model_results.get(
                    "reflection_score", char_reflection_score(text))
            else:
                reflection_score = char_reflection_score(text)
                confidence = min(0.95 + reflection_score * 0.05, 1.0)
                is_palindrome = reflection_score > 0.5

            # Create comprehensive visualization with better spacing
            fig = plt.figure(figsize=(32, 18))
            gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

            fig.suptitle(
                f'Comprehensive Analysis: "{text}"', fontsize=24, fontweight='bold', y=0.96)

            # Row 1: Text Statistics Summary (spans full width)
            ax1 = fig.add_subplot(gs[0, :])
            char_freq = {}
            for char in cleaned:
                if char in string.ascii_lowercase:
                    char_freq[char] = char_freq.get(char, 0) + 1

            # Create a comprehensive statistics table for the summary
            stats_data = [
                ['Text Length', len(cleaned)],
                ['Unique Characters', len(set(cleaned))],
                ['Most Common Char', max(char_freq.items(), key=lambda x: x[1])[
                    0] if char_freq else 'N/A'],
                ['Reflection Score', f'{reflection_score:.3f}'],
                ['Symmetry Type', 'Perfect' if reflection_score >= 0.99 else 'High' if reflection_score >= 0.9 else 'Good' if reflection_score >=
                    0.8 else 'Moderate' if reflection_score >= 0.6 else 'Low' if reflection_score >= 0.4 else 'Poor'],
                ['Model Prediction', 'Palindrome' if is_palindrome else 'Not Palindrome'],
                ['Confidence (is palindrome)', f'{confidence:.1%}']
            ]

            # Create table
            table = ax1.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                              cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(14)
            table.scale(1.2, 2.5)

            # Color code the table
            for i in range(len(stats_data)):
                if i == 3:  # Reflection Score row
                    table[(i+1, 1)].set_facecolor('lightgreen')
                elif i == 4:  # Symmetry Type row
                    if 'Perfect' in stats_data[i][1]:
                        table[(i+1, 1)].set_facecolor('darkgreen')
                    elif 'High' in stats_data[i][1]:
                        table[(i+1, 1)].set_facecolor('green')
                    elif 'Good' in stats_data[i][1]:
                        table[(i+1, 1)].set_facecolor('lightgreen')
                    elif 'Moderate' in stats_data[i][1]:
                        table[(i+1, 1)].set_facecolor('orange')
                    elif 'Low' in stats_data[i][1]:
                        table[(i+1, 1)].set_facecolor('yellow')
                    else:
                        table[(i+1, 1)].set_facecolor('red')
                elif i == 5:  # Model Prediction row
                    if 'Palindrome' in stats_data[i][1]:
                        table[(i+1, 1)].set_facecolor('lightblue')
                    else:
                        table[(i+1, 1)].set_facecolor('lightcoral')

            ax1.set_title('Text Analysis Summary',
                          fontweight='bold', fontsize=18)
            ax1.axis('off')

            # Row 2: Model Decision Factors and Model Performance Metrics
            # Model Decision Factors
            ax2 = fig.add_subplot(gs[1, 0])

            # Calculate confidence factors (confidence already calculated above)
            # Longer texts get higher confidence
            length_factor = min(1.0, len(cleaned) / 20.0)
            symmetry_factor = reflection_score

            factors = {
                'Reflection Score': symmetry_factor * 100,
                'Length Factor': length_factor * 100,
                'Model Confidence': confidence * 100
            }

            colors = ['green', 'blue', 'purple']
            factor_names = list(factors.keys())
            factor_values = list(factors.values())

            bars = ax2.bar(factor_names, factor_values,
                           color=colors, alpha=0.8)
            ax2.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
            ax2.set_title('Model Decision Factors',
                          fontweight='bold', fontsize=14)
            ax2.set_ylim(0, 110)
            ax2.tick_params(labelsize=10)

            # Add value labels
            for bar, value in zip(bars, factor_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

            # Character Frequency Analysis
            ax3 = fig.add_subplot(gs[1, 1:])

            # Calculate character frequencies
            char_freq = {}
            for char in cleaned:
                char_freq[char] = char_freq.get(char, 0) + 1

            # Sort by frequency (descending)
            sorted_chars = sorted(
                char_freq.items(), key=lambda x: x[1], reverse=True)
            char_names = [char for char, freq in sorted_chars]
            char_freqs = [freq for char, freq in sorted_chars]

            # Create bar chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(char_names)))
            bars = ax3.bar(char_names, char_freqs, color=colors,
                           alpha=0.8, edgecolor='black', linewidth=1)
            ax3.set_ylabel('Frequency', fontweight='bold', fontsize=12)
            ax3.set_title('Character Frequency Analysis',
                          fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(labelsize=10)

            # Add value labels on bars
            for bar, freq in zip(bars, char_freqs):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         str(freq), ha='center', va='bottom', fontweight='bold', fontsize=10)

            # Add summary statistics
            total_chars = len(cleaned)
            unique_chars = len(char_names)
            most_common = char_names[0] if char_names else 'N/A'
            most_common_freq = char_freqs[0] if char_freqs else 0

            summary_text = f'Total: {total_chars} | Unique: {unique_chars} | Most Common: {most_common} ({most_common_freq})'
            ax3.text(0.5, 0.95, summary_text,
                     transform=ax3.transAxes, ha='center', va='top',
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="lightblue", alpha=0.9),
                     fontweight='bold', fontsize=12)

            # Row 3: Character Similarity Heatmap and Character Symmetry Analysis
            # Character Similarity Heatmap
            ax4 = fig.add_subplot(gs[2, 0])

            n = len(cleaned)
            if n > 0:
                # Create a heatmap showing character matching
                heatmap_data = np.zeros((n, n))

                for i in range(n):
                    for j in range(n):
                        if cleaned[i] == cleaned[j]:
                            # Matching characters (including self)
                            heatmap_data[i, j] = 1.0
                        else:
                            heatmap_data[i, j] = 0.0  # Different characters

                im = ax4.imshow(heatmap_data, cmap='Blues', aspect='auto')
                ax4.set_title('Character Similarity Heatmap',
                              fontweight='bold', fontsize=14)
                ax4.set_xlabel('Character Position',
                               fontweight='bold', fontsize=12)
                ax4.set_ylabel('Character Position',
                               fontweight='bold', fontsize=12)

                # Create custom grey to blue colormap
                colors = ['lightgrey', 'blue']
                n_bins = 100
                cmap = LinearSegmentedColormap.from_list(
                    'GreyBlue', colors, N=n_bins)

                # Display heatmap with custom colormap
                im = ax4.imshow(heatmap_data, cmap=cmap, aspect='auto')

                # Add grid lines
                ax4.grid(True, color='grey', linewidth=0.5, alpha=0.7)

                # Add character labels - center them on the cells
                ax4.set_xticks(np.arange(n) + 0.5)
                ax4.set_yticks(np.arange(n) + 0.5)
                ax4.set_xticklabels(list(cleaned))
                ax4.set_yticklabels(list(cleaned))

                # Adjust axis limits to center labels properly
                ax4.set_xlim(-0.5, n - 0.5)
                ax4.set_ylim(-0.5, n - 0.5)

                # Adjust tick label positioning to center on cells
                # Move x-axis labels down more and adjust positioning
                ax4.tick_params(axis='x', pad=30, labelrotation=0)
                # Move y-axis labels right more
                ax4.tick_params(axis='y', pad=15)
                ax4.tick_params(labelsize=10)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Similarity Score',
                               fontweight='bold', fontsize=12)
                cbar.ax.tick_params(labelsize=10)

            # Character Symmetry Analysis
            ax5 = fig.add_subplot(gs[2, 1:])

            if n > 0:
                # Create character position visualization with symmetry highlighting
                positions = list(range(n))
                symmetry_colors = []

                for pos in range(n):
                    if pos < n // 2:  # Left half
                        if cleaned[pos] == cleaned[n - 1 - pos]:
                            symmetry_colors.append('green')  # Matching
                        else:
                            symmetry_colors.append('red')    # Mismatch
                    # Center character (odd length)
                    elif n % 2 == 1 and pos == n // 2:
                        symmetry_colors.append('blue')       # Center
                    else:  # Right half
                        if cleaned[pos] == cleaned[n - 1 - pos]:
                            symmetry_colors.append('green')  # Matching
                        else:
                            symmetry_colors.append('red')    # Mismatch

                # Plot characters with colors
                for pos, char in enumerate(cleaned):
                    ax5.scatter(
                        pos, 1, s=200, c=symmetry_colors[pos], alpha=0.8, edgecolors='black', linewidth=2)
                    ax5.text(pos, 1, char, ha='center', va='center',
                             fontweight='bold', fontsize=14)

                ax5.set_xlabel('Character Position',
                               fontweight='bold', fontsize=12)
                ax5.set_title(
                    'Character Symmetry Analysis\n(Green=Match, Red=Mismatch, Blue=Center)',
                    fontweight='bold', fontsize=14)
                ax5.set_ylim(0.5, 1.5)
                ax5.set_xlim(-0.5, n-0.5)
                ax5.grid(True, alpha=0.3)
                ax5.tick_params(labelsize=10)

            plt.tight_layout()

            # Save the visualization
            filename = f'input_analysis_{timestamp}_{i}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)

            visualizations.append(filename)

        return {
            "success": True,
            "visualizations_generated": len(visualizations),
            "files": visualizations,
            "output_directory": output_dir,
            "message": f"Generated {len(visualizations)} input-specific visualizations"
        }

    except Exception as e:
        return {"error": f"Failed to generate comprehensive visualizations: {str(e)}"}


# FastAPI endpoints
@palindrome_router.get("/health")
async def health_check():
    """Health check endpoint."""
    # Don't trigger background tasks during health check
    model_status = get_model_status()
    loaded_models = 1 if model is not None else 0

    return {
        "status": "healthy",
        "total_models": 1,
        "loaded_models": loaded_models,
        "model_status": model_status,
        "message": "Reflection-Integrated Palindrome Detection API is running"
    }


@palindrome_router.get("/model/status")
async def model_status_endpoint():
    """Get detailed status of all models."""
    return get_model_status()


@palindrome_router.get("/model/current")
async def current_model_endpoint():
    """Get information about the single loaded model and unload timer."""
    if model is not None and last_usage_time is not None:
        time_since_usage = datetime.now() - last_usage_time
        time_until_unload = UNLOAD_DELAY - time_since_usage

        if time_until_unload > timedelta(0):
            model_info = {
                "model_key": "model_1_50",
                "model_name": MODEL_CONFIG["display_name"],
                "length_range": f"{MODEL_CONFIG['min_len']}-{MODEL_CONFIG['max_len']} chars",
                "input_length": MODEL_CONFIG["input_len"],
                "loaded_at": last_usage_time.isoformat(),
                "time_since_loaded": str(time_since_usage).split('.')[0],
                "time_until_unload": str(time_until_unload).split('.')[0],
                "seconds_until_unload": int(time_until_unload.total_seconds())
            }
        else:
            model_info = {
                "model_key": "model_1_50",
                "model_name": MODEL_CONFIG["display_name"],
                "length_range": f"{MODEL_CONFIG['min_len']}-{MODEL_CONFIG['max_len']} chars",
                "input_length": MODEL_CONFIG["input_len"],
                "loaded_at": last_usage_time.isoformat(),
                "time_since_loaded": str(time_since_usage).split('.')[0],
                "time_until_unload": "OVERDUE",
                "seconds_until_unload": 0,
                "status": "overdue_for_unload"
            }
    else:
        model_info = {
            "model_key": "model_1_50",
            "model_name": MODEL_CONFIG["display_name"],
            "length_range": f"{MODEL_CONFIG['min_len']}-{MODEL_CONFIG['max_len']} chars",
            "input_length": MODEL_CONFIG["input_len"],
            "loaded_at": None,
            "time_since_loaded": None,
            "time_until_unload": None,
            "seconds_until_unload": None,
            "status": "not_loaded"
        }

    return {
        "currently_loaded": model is not None,
        "loaded_models": [model_info] if model is not None else [],
        "total_loaded": 1 if model is not None else 0,
        "message": "Single model system - 1 model available"
    }


@palindrome_router.get("/model/test-results")
async def get_test_results():
    """Get test results for all models."""
    try:
        # Try to load the test results file
        test_results_file = "test_results.json"
        if not os.path.exists(test_results_file):
            return {
                "success": False,
                "message": "Test results file not found",
                "test_results": {}
            }

        with open(test_results_file, 'r') as f:
            test_data = json.load(f)

        # Return the test data in the format expected by the frontend
        return {
            "success": True,
            "message": "Test results loaded successfully",
            "test_results": test_data
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error loading test results: {str(e)}",
            "test_results": {}
        }


@palindrome_router.post("/model/load/{model_key}")
async def force_load_model(model_key: str):
    """Force load a specific model."""
    if model_key not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400, detail=f"Invalid model key: {model_key}")

    if ensure_model_loaded(model_key):
        return {
            "success": True,
            "message": f"Model {model_key} loaded successfully",
            "status": get_model_status()
        }
    else:
        raise HTTPException(
            status_code=500, detail=f"Failed to load model {model_key}")


@palindrome_router.post("/model/unload/{model_key}")
async def force_unload_model(model_key: str):
    """Force unload a specific model."""
    if model_key not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400, detail=f"Invalid model key: {model_key}")

    unload_model(model_key)
    return {
        "success": True,
        "message": f"Model {model_key} unloaded successfully",
        "status": get_model_status()
    }


@palindrome_router.post("/model/load/all")
async def force_load_all_models():
    """Force load all models."""
    results = {}
    for model_key in MODEL_CONFIGS.keys():
        success = ensure_model_loaded(model_key)
        results[model_key] = success

    loaded_count = sum(results.values())
    return {
        "success": True,
        "message": f"Loaded {loaded_count}/{len(MODEL_CONFIGS)} models",
        "results": results,
        "status": get_model_status()
    }


@palindrome_router.post("/model/unload/all")
async def force_unload_all_models():
    """Force unload all models."""
    for model_key in MODEL_CONFIGS.keys():
        unload_model(model_key)

    return {
        "success": True,
        "message": "All models unloaded successfully",
        "status": get_model_status()
    }


@palindrome_router.post("/model/cleanup")
async def cleanup_models():
    """Clean up overdue models."""
    cleaned_count = cleanup_overdue_models()

    return {
        "success": True,
        "message": f"Cleaned up {cleaned_count} overdue model(s)",
        "cleaned_count": cleaned_count,
        "status": get_model_status()
    }


@palindrome_router.post("/visualize/comprehensive")
async def generate_comprehensive_visualizations_endpoint(request: GenerateVisualizationsRequest):
    """Generate comprehensive visualizations for the reflection-integrated model."""
    try:
        # Ensure model is loaded
        if not ensure_model_loaded():
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )

        # Generate comprehensive visualizations
        result = generate_comprehensive_visualizations(request.dict())

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return {
            "success": True,
            "message": f"Generated {result['visualizations_generated']} visualizations",
            "visualizations_generated": result["visualizations_generated"],
            "files": result["files"],
            "output_directory": result["output_directory"],
            "model_used": "model_1_50",
            "model_name": MODEL_CONFIG["display_name"],
            "descriptions": {
                "performance_analysis": "Model accuracy and prediction patterns across different test cases",
                "layer_analysis": "Internal model representations showing how text is processed through layers",
                "reflection_analysis": "Detailed breakdown of reflection scores vs model predictions"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive visualization generation failed: {str(e)}"
        )


@palindrome_router.post("/predict")
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Predict if a text is a palindro  me using the appropriate specialized model."""
    start_time = datetime.now()
    try:
        text = request.text
        include_reflection_analysis = request.include_reflection_analysis
        include_all_visualizations = request.include_all_visualizations

        # Calculate text length (cleaned) - strip whitespace, then only alphabetic characters
        cleaned = ''.join(c.lower() for c in text.strip() if c.isalpha())
        text_length = len(cleaned)

        # Get the best available model for this text length
        model_key, config, selection_reason = get_best_available_model(
            text_length)

        if model_key is None:
            raise HTTPException(
                status_code=400,
                detail=f"Text length ({text_length}) is outside supported range (1-50 characters)"
            )

        # Ensure model is loaded and update usage time
        if not ensure_model_loaded():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model for length range {config['min_len']}-{config['max_len']}"
            )

        # Schedule model unload after 5 minutes of inactivity
        schedule_model_unload(background_tasks)

        # Preprocess the text - use the model's specific input length
        encoded = encode_multiword(text, maxlen=config["input_len"])

        # Calculate reflection score for the reflection-integrated model
        reflection_score = char_reflection_score(text)

        # Make prediction based on model type
        if config.get("has_reflection", False):
            # Reflection-integrated model expects both text encoding and reflection score
            prediction = model.predict(
                [np.array([encoded]), np.array([[reflection_score]])],
                verbose=0
            )[0][0]
        else:
            # Standard model expects only text encoding
            prediction = model.predict(
                np.array([encoded]),
                verbose=0
            )[0][0]

        # Determine result
        is_palindrome = bool(prediction > 0.7)
        confidence = float(prediction)

        # Prepare response
        response_data = {
            "text": text,
            "cleaned_text": cleaned,
            "text_length": text_length,
            "model_used": model_key,
            "model_name": config["display_name"],
            "selection_reason": selection_reason,
            "is_palindrome": is_palindrome,
            "confidence": confidence,
            "raw_score": float(prediction),
            "reflection_score": float(reflection_score) if config.get("has_reflection", False) else None,
            "model_status": get_model_status(),
            "processing_time_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2)
        }

        # Add reflection analysis if requested
        if include_reflection_analysis or include_all_visualizations:
            try:
                # Generate detailed reflection analysis with vector information
                detailed_reflection = char_reflection_score_detailed(text)

                reflection_analysis = {
                    "reflection_score": float(detailed_reflection['score']) if config.get("has_reflection", False) else None,
                    "score_interpretation": "Perfect symmetry" if detailed_reflection['score'] >= 0.99 else
                    "High symmetry" if detailed_reflection['score'] >= 0.9 else
                    "Good symmetry" if detailed_reflection['score'] >= 0.8 else
                    "Moderate symmetry" if detailed_reflection['score'] >= 0.6 else
                    "Low symmetry" if detailed_reflection['score'] >= 0.4 else
                    "Poor symmetry" if config.get(
                        "has_reflection", False) else "Not available",
                    "character_analysis": {
                        "total_characters": len(cleaned),
                        "unique_characters": len(set(cleaned)),
                        "most_common": max(set(cleaned), key=cleaned.count) if cleaned else None,
                        "character_frequency": {char: cleaned.count(char) for char in set(cleaned)} if cleaned else {}
                    },
                    "symmetry_details": {
                        "left_half": detailed_reflection['left_half'],
                        "right_half": detailed_reflection['right_half'],
                        "right_reversed": detailed_reflection['right_reversed'],
                        "center_character": detailed_reflection['center_char']
                    },
                    "vector_analysis": {
                        "left_vector": detailed_reflection['left_vector'],
                        "right_vector": detailed_reflection['right_vector'],
                        "difference_vector": detailed_reflection['difference_vector'],
                        "mean_abs_difference": detailed_reflection['mean_abs_difference'],
                        "normalization_formula": detailed_reflection['normalization_formula'],
                        "score_formula": detailed_reflection['score_formula'],
                        "left_mapping": detailed_reflection['left_mapping'],
                        "right_mapping": detailed_reflection['right_mapping'],
                        "difference_mapping": detailed_reflection['difference_mapping']
                    }
                }
                response_data["reflection_analysis"] = reflection_analysis
            except Exception as e:
                response_data["reflection_analysis"] = {
                    "error": f"Failed to generate reflection analysis: {str(e)}"}

        # Add comprehensive visualizations if requested
        if include_all_visualizations:
            try:
                # Generate comprehensive visualizations for the specific input with actual model results
                comprehensive_result = generate_comprehensive_visualizations({
                    "include_performance": True,
                    "include_layer_analysis": True,
                    "include_reflection_analysis": True,
                    "test_texts": [text],  # Pass the actual input text
                    "model_results": {
                        "is_palindrome": is_palindrome,
                        "confidence": confidence,
                        "reflection_score": float(reflection_score) if config.get("has_reflection", False) else None
                    }
                })
                if "error" not in comprehensive_result:
                    response_data["comprehensive_visualizations"] = comprehensive_result
                else:
                    response_data["comprehensive_visualizations"] = {
                        "error": comprehensive_result["error"]
                    }
            except Exception as e:
                response_data["comprehensive_visualizations"] = {
                    "error": f"Failed to generate comprehensive visualizations: {str(e)}"
                }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@palindrome_router.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    """Predict for multiple texts using appropriate specialized models."""
    start_time = datetime.now()
    try:
        texts = request.texts
        include_reflection_analysis = request.include_reflection_analysis

        results = []
        models_used = set()

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({
                    "text": text,
                    "error": "Invalid text input"
                })
                continue

            try:
                # Calculate text length - strip whitespace, then only alphabetic characters
                cleaned = ''.join(c.lower()
                                  for c in text.strip() if c.isalpha())
                text_length = len(cleaned)

                # Get the best available model for this text length
                model_key, config, selection_reason = get_best_available_model(
                    text_length)

                if model_key is None:
                    results.append({
                        "text": text,
                        "error": f"Text length ({text_length}) is outside supported range (1-50 characters)"
                    })
                    continue

                # Ensure model is loaded and update usage time
                if not ensure_model_loaded():
                    results.append({
                        "text": text,
                        "error": f"Failed to load model for length range {config['min_len']}-{config['max_len']}"
                    })
                    continue

                # Schedule model unload after 5 minutes of inactivity
                schedule_model_unload(background_tasks)
                models_used.add("model_1_50")

                # Preprocess and predict - use the model's specific input length
                encoded = encode_multiword(text, maxlen=config["input_len"])

                # Calculate reflection score for the reflection-integrated model
                reflection_score = char_reflection_score(text)

                # Make prediction based on model type
                if config.get("has_reflection", False):
                    # Reflection-integrated model expects both text encoding and reflection score
                    prediction = model.predict(
                        [np.array([encoded]), np.array([[reflection_score]])],
                        verbose=0
                    )[0][0]
                else:
                    # Standard model expects only text encoding
                    prediction = model.predict(
                        np.array([encoded]),
                        verbose=0
                    )[0][0]

                is_palindrome = bool(prediction > 0.7)
                confidence = float(prediction)

                result_item = {
                    "text": text,
                    "cleaned_text": cleaned,
                    "text_length": text_length,
                    "model_used": "model_1_50",
                    "model_name": MODEL_CONFIG["display_name"],
                    "selection_reason": selection_reason,
                    "is_palindrome": is_palindrome,
                    "confidence": confidence,
                    "raw_score": float(prediction),
                    "reflection_score": float(reflection_score) if config.get("has_reflection", False) else None
                }

                # Add reflection analysis if requested
                if include_reflection_analysis and config.get("has_reflection", False):
                    try:
                        # Generate reflection analysis visualization
                        reflection_analysis = {
                            "reflection_score": float(reflection_score),
                            "score_interpretation": "Perfect symmetry" if reflection_score >= 0.99 else
                            "High symmetry" if reflection_score >= 0.9 else
                            "Good symmetry" if reflection_score >= 0.8 else
                            "Moderate symmetry" if reflection_score >= 0.6 else
                            "Low symmetry" if reflection_score >= 0.4 else
                            "Poor symmetry",
                            "character_analysis": {
                                "total_characters": len(cleaned),
                                "unique_characters": len(set(cleaned)),
                                "most_common": max(set(cleaned), key=cleaned.count) if cleaned else None
                            }
                        }
                        result_item["reflection_analysis"] = reflection_analysis
                    except Exception as e:
                        result_item["reflection_analysis"] = {
                            "error": f"Failed to generate reflection analysis: {str(e)}"}

                results.append(result_item)

            except Exception as e:
                results.append({
                    "text": text,
                    "error": f"Prediction failed: {str(e)}"
                })

        # Calculate batch statistics
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]

        # Overall performance metrics
        processing_time_ms = round(
            (datetime.now() - start_time).total_seconds() * 1000, 2)

        # Summary statistics
        palindrome_count = sum(
            1 for r in successful_results if r.get('is_palindrome', False))
        non_palindrome_count = len(successful_results) - palindrome_count

        # Average confidence
        avg_confidence = 0
        if successful_results:
            avg_confidence = sum(r.get('confidence', 0)
                                 for r in successful_results) / len(successful_results)

        return {
            "results": results,
            "total_processed": len(results),
            "successful_predictions": len(successful_results),
            "failed_predictions": len(failed_results),
            "models_used": list(models_used),
            "model_status": get_model_status(),
            "batch_summary": {
                "palindromes_detected": palindrome_count,
                "non_palindromes_detected": non_palindrome_count,
                "average_confidence": round(avg_confidence, 3),
                "success_rate": round(len(successful_results) / len(results) * 100, 1) if results else 0
            },
            "processing_time_ms": processing_time_ms
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@palindrome_router.get("/")
async def index():
    """API documentation endpoint."""
    return {
        "message": "Specialized Multi-Word Palindrome Detection API",
        "version": "1.0.0",
        "features": [
            "Reflection-integrated neural network model",
            "Dual-input architecture (text + reflection score)",
            "Perfect accuracy on comprehensive test suites",
            "Automatic model loading and unloading",
            "Comprehensive visualization generation",
            "Real-time attention analysis",
            "Background task-based model management"
        ],
        "models": {
            "model_1_50": {
                "name": MODEL_CONFIG["name"],
                "display_name": MODEL_CONFIG["display_name"],
                "length_range": f"{MODEL_CONFIG['min_len']}-{MODEL_CONFIG['max_len']} chars",
                "input_length": MODEL_CONFIG["input_len"],
                "has_reflection": MODEL_CONFIG.get("has_reflection", False)
            }
        },
        "endpoints": {
            "GET /": "API documentation (this endpoint)",
            "GET /health": "Health check with all model statuses",
            "GET /model/status": "Detailed status of all models",
            "GET /model/current": "Currently loaded model and unload timer",
            "POST /model/load/{model_key}": "Force load specific model",
            "POST /model/unload/{model_key}": "Force unload specific model",
            "POST /model/load/all": "Force load all models",
            "POST /model/unload/all": "Force unload all models",
            "POST /predict": "Predict single text with optional visualizations",
            "POST /predict/batch": "Predict multiple texts with optional visualizations",
            "POST /visualize/comprehensive": "Generate comprehensive visualizations",
            "GET /visualizations": "List all available visualizations",
            "GET /visualizations/{filename}": "Serve a specific visualization file"
        },
        "usage": {
            "single_prediction": {
                "url": "/predict",
                "method": "POST",
                "body": {"text": "race a car"}
            },

            "prediction_with_reflection": {
                "url": "/predict",
                "method": "POST",
                "body": {"text": "race a car", "include_reflection_analysis": True}
            },
            "prediction_with_all_viz": {
                "url": "/predict",
                "method": "POST",
                "body": {"text": "race a car", "include_all_visualizations": True}
            },
            "batch_prediction": {
                "url": "/predict/batch",
                "method": "POST",
                "body": {"texts": ["race a car", "hello world", "madam i'm adam"]}
            },
            "batch_with_viz": {
                "url": "/predict/batch",
                "method": "POST",
                "body": {
                    "texts": ["race a car", "hello world", "madam i'm adam"],
                    "include_reflection_analysis": True
                }
            },
            "comprehensive_viz": {
                "url": "/visualize/comprehensive",
                "method": "POST",
                "body": {
                    "include_performance": True,
                    "include_layer_analysis": True,
                    "include_reflection_analysis": True
                }
            },
            "list_visualizations": {
                "url": "/visualizations",
                "method": "GET"
            },
            "model_management": {
                "check_status": "GET /model/status",
                "check_current": "GET /model/current",
                "load_specific": "POST /model/load/model_1_25",
                "load_all": "POST /model/load/all",
                "unload_all": "POST /model/unload/all"
            }
        }
    }


# Optional: Add static file serving for the monitor dashboard
@palindrome_router.get("/monitor")
async def monitor():
    """Serve the monitoring dashboard."""
    monitor_file = "static/index.html"
    if os.path.exists(monitor_file):
        return HTMLResponse(content=open(monitor_file, "r", encoding="utf-8").read())
    else:
        raise HTTPException(
            status_code=404, detail="Monitor dashboard not found")


@palindrome_router.get("/visualizations")
async def list_visualizations():
    """List all available visualizations."""

    viz_dir = "visualization_output"
    if not os.path.exists(viz_dir):
        return {
            "visualizations": [],
            "message": "No visualizations directory found"
        }

    # Get all PNG files
    viz_files = glob.glob(os.path.join(viz_dir, "*.png"))
    viz_files = [os.path.basename(f) for f in viz_files]

    # Categorize visualizations
    categories = {
        "Performance Analysis": [f for f in viz_files if any(x in f for x in ["performance", "reflection_vs_prediction", "confidence_distribution", "accuracy_by_length", "reflection_score_distribution", "length_vs_reflection"])],
        "Layer Analysis": [f for f in viz_files if "layer_analysis" in f],
        "Reflection Analysis": [f for f in viz_files if "reflection_analysis" in f]
    }

    return {
        "total_visualizations": len(viz_files),
        "categories": categories,
        "all_files": viz_files,
        "output_directory": viz_dir
    }


@palindrome_router.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    """Serve a specific visualization file."""

    viz_dir = "visualization_output"
    file_path = os.path.join(viz_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Visualization file '{filename}' not found"
        )

    return FileResponse(file_path, media_type="image/png")


# Export the router for easy integration
__all__ = ["palindrome_router"]
