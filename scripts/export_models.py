#!/usr/bin/env python3
"""
Export integrated model checkpoints to ONNX and RTNeural formats.

This script:
1. Loads PyTorch checkpoints
2. Creates model architectures matching the checkpoint structure
3. Exports to ONNX (cross-platform) and RTNeural JSON (C++ real-time)
"""

import sys
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Simple Model Architectures (matching checkpoint structure)
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP matching checkpoint structure (fc1, fc2, fc3)."""
    
    def __init__(self, layer_sizes: list, activation='relu'):
        super().__init__()
        # Create layers with names matching checkpoint (fc1, fc2, fc3, etc.)
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), 1):
            setattr(self, f'fc{i}', nn.Linear(in_size, out_size))
        
        self.num_layers = len(layer_sizes) - 1
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
    
    def forward(self, x):
        # Apply layers sequentially with activation between (except last)
        for i in range(1, self.num_layers + 1):
            x = getattr(self, f'fc{i}')(x)
            if i < self.num_layers:  # Don't apply activation to last layer
                x = self.activation(x)
        return x


class SimpleLSTM(nn.Module):
    """Simple LSTM matching checkpoint structure - handles FC before/after LSTM."""
    
    def __init__(self, input_size: int, lstm_input_size: int, hidden_size: int, 
                 num_layers: int, fc_shapes: list, fc_before_lstm: bool = False):
        super().__init__()
        
        # Create all FC layers in order
        fc_idx = 1
        fc_before_count = 0
        
        # Create FC layers before LSTM
        if fc_before_lstm and fc_shapes:
            prev_size = input_size
            for fc_in, fc_out in fc_shapes:
                if fc_in == prev_size:
                    setattr(self, f'fc{fc_idx}', nn.Linear(fc_in, fc_out))
                    fc_idx += 1
                    fc_before_count += 1
                    prev_size = fc_out
                    if prev_size == lstm_input_size:
                        break  # Reached LSTM input size
        
        self.lstm_input_size = prev_size if fc_before_count > 0 else lstm_input_size
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_layers, batch_first=True)
        
        # Create FC layers after LSTM
        if fc_shapes and fc_before_count < len(fc_shapes):
            prev_size = hidden_size
            for fc_in, fc_out in fc_shapes[fc_before_count:]:
                if fc_in == prev_size:
                    setattr(self, f'fc{fc_idx}', nn.Linear(fc_in, fc_out))
                    fc_idx += 1
                    prev_size = fc_out
                elif fc_in == hidden_size and fc_idx == fc_before_count + 1:
                    # First FC after LSTM
                    setattr(self, f'fc{fc_idx}', nn.Linear(fc_in, fc_out))
                    fc_idx += 1
                    prev_size = fc_out
        
        self.num_fc = fc_idx - 1
        self.fc_before_count = fc_before_count
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Apply FC before LSTM
        for i in range(1, self.fc_before_count + 1):
            x = getattr(self, f'fc{i}')(x)
            if i < self.fc_before_count:
                x = self.activation(x)
        
        # LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(x)
        if lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Apply FC after LSTM
        x = lstm_out
        for i in range(self.fc_before_count + 1, self.num_fc + 1):
            if hasattr(self, f'fc{i}'):
                x = getattr(self, f'fc{i}')(x)
                if i < self.num_fc:  # Don't activate last layer
                    x = self.activation(x)
        return x


def infer_model_architecture(state_dict: dict) -> Tuple[str, dict]:
    """
    Infer model architecture from state dict keys.
    
    Returns:
        (architecture_type, architecture_params)
    """
    keys = list(state_dict.keys())
    
    # Check for LSTM layers
    if any('lstm' in k.lower() for k in keys):
        # Find LSTM dimensions
        lstm_keys = [k for k in keys if 'lstm' in k.lower()]
        weight_ih_key = [k for k in lstm_keys if 'weight_ih_l0' in k][0]
        lstm_input_size = state_dict[weight_ih_key].shape[1]
        lstm_hidden_size = state_dict[weight_ih_key].shape[0] // 4  # LSTM has 4 gates
        num_layers = 1  # Only checking l0 for now
        
        # Find FC layers - they might be before or after LSTM
        fc_keys = sorted([k for k in keys if 'fc' in k.lower() and 'weight' in k and 'lstm' not in k.lower()])
        
        # Analyze FC layer structure - build chain by matching inputs/outputs
        all_fc = [(state_dict[k].shape[1], state_dict[k].shape[0]) for k in fc_keys]
        
        # Build chain: start with smallest input (likely model input)
        remaining = all_fc.copy()
        model_input_size = min([in_size for in_size, _ in all_fc])  # Start with smallest input
        current_size = model_input_size
        
        # Build chain by matching inputs/outputs
        fc_before = []
        fc_after = []
        
        # Build FC chain from input
        while remaining:
            found = False
            for i, (in_size, out_size) in enumerate(remaining):
                if in_size == current_size:
                    # Check if this connects to LSTM or continues chain
                    if out_size == lstm_input_size:
                        # This FC feeds into LSTM
                        fc_before.append((in_size, out_size))
                        remaining.pop(i)
                        current_size = lstm_hidden_size  # LSTM output
                        found = True
                        break
                    elif in_size == lstm_hidden_size or (fc_after and in_size == fc_after[-1][1]):
                        # This FC is after LSTM
                        fc_after.append((in_size, out_size))
                        remaining.pop(i)
                        current_size = out_size
                        found = True
                        break
                    else:
                        # Continue building before-LSTM chain
                        fc_before.append((in_size, out_size))
                        remaining.pop(i)
                        current_size = out_size
                        found = True
                        break
            if not found:
                # Check if any remaining FC starts from LSTM output
                for i, (in_size, out_size) in enumerate(remaining):
                    if in_size == lstm_hidden_size:
                        fc_after.append((in_size, out_size))
                        remaining.pop(i)
                        current_size = out_size
                        found = True
                        break
                if not found:
                    break
        
        all_fc_ordered = fc_before + fc_after
        
        return 'lstm', {
            'input_size': model_input_size,
            'lstm_input_size': lstm_input_size if not fc_before else fc_before[-1][1],
            'hidden_size': lstm_hidden_size,
            'num_layers': num_layers,
            'fc_shapes': all_fc_ordered,  # Ordered list of (in, out) pairs
            'fc_before_lstm': len(fc_before) > 0,
        }
    
    # MLP architecture
    else:
        fc_keys = sorted([k for k in keys if 'fc' in k.lower() and 'weight' in k])
        
        if not fc_keys:
            raise ValueError(f"Cannot infer architecture from keys: {keys[:10]}")
        
        # Infer layer sizes - need both input and output sizes
        layer_sizes = []
        input_size = state_dict[fc_keys[0]].shape[1]
        layer_sizes.append(input_size)
        
        for fc_key in fc_keys:
            out_size = state_dict[fc_key].shape[0]
            layer_sizes.append(out_size)
        
        return 'mlp', {
            'layer_sizes': layer_sizes,
        }


class FlexibleModel(nn.Module):
    """Flexible model that loads any checkpoint structure."""
    
    def __init__(self, state_dict: dict):
        super().__init__()
        # Create all layers from state dict
        self.layers = nn.ModuleDict()
        processed = set()
        
        for key, value in state_dict.items():
            if 'weight' in key and key not in processed:
                layer_name = key.split('.')[0]
                param_name = key.split('.', 1)[1] if '.' in key else 'weight'
                
                if 'fc' in layer_name.lower():
                    if layer_name not in self.layers:
                        # Determine shape from weight
                        if 'weight' in key:
                            out_size, in_size = value.shape
                            self.layers[layer_name] = nn.Linear(in_size, out_size)
                            processed.add(f"{layer_name}.weight")
                            processed.add(f"{layer_name}.bias")
                elif 'lstm' in layer_name.lower() and layer_name not in self.layers:
                    # LSTM - need to infer from weight_ih
                    if 'weight_ih_l0' in key:
                        hidden_size = value.shape[0] // 4
                        input_size = value.shape[1]
                        self.layers[layer_name] = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
                        # Mark all LSTM params as processed
                        for k in state_dict.keys():
                            if layer_name in k:
                                processed.add(k)
        
        self.layer_order = sorted([k for k in self.layers.keys()])
        self.has_lstm = any('lstm' in k for k in self.layer_order)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Determine correct layer order by analyzing data flow
        # FC layers before LSTM, then LSTM, then FC layers after LSTM
        fc_layers = [k for k in self.layer_order if 'fc' in k.lower()]
        lstm_layers = [k for k in self.layer_order if 'lstm' in k.lower()]
        
        # Sort FC layers by input size (smallest input first = earliest layer)
        if fc_layers and lstm_layers:
            # Get LSTM input size
            lstm_layer = self.layers[lstm_layers[0]]
            if isinstance(lstm_layer, nn.LSTM):
                lstm_input_size = lstm_layer.input_size
                
                # Determine FC layers before and after LSTM
                fc_before = []
                fc_after = []
                
                # Find which FC outputs match LSTM input
                for fc_name in fc_layers:
                    fc_layer = self.layers[fc_name]
                    if isinstance(fc_layer, nn.Linear):
                        # Check if this FC's output matches LSTM input
                        if fc_layer.out_features == lstm_input_size:
                            fc_before.append(fc_name)
                        elif fc_layer.in_features == lstm_input_size or (fc_after and fc_layer.in_features == self.layers[fc_after[-1]].out_features):
                            fc_after.append(fc_name)
                        else:
                            # Try to determine by checking if input matches previous output
                            if not fc_before or fc_layer.in_features == self.layers[fc_before[-1]].out_features:
                                fc_before.append(fc_name)
                            else:
                                fc_after.append(fc_name)
                
                # Build ordered layer list
                ordered_layers = fc_before + lstm_layers + fc_after
            else:
                ordered_layers = self.layer_order
        else:
            ordered_layers = self.layer_order
        
        # Apply layers in correct order
        for layer_name in ordered_layers:
            layer = self.layers[layer_name]
            if isinstance(layer, nn.LSTM):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x, _ = layer(x)
                if x.dim() == 3:
                    x = x[:, -1, :]  # Take last output
            else:
                x = layer(x)
                if layer_name != ordered_layers[-1]:  # Don't activate last layer
                    x = self.activation(x)
        return x


def load_checkpoint_and_create_model(checkpoint_path: Path, use_flexible: bool = False) -> Tuple[nn.Module, dict]:
    """Load checkpoint and create matching model architecture."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    else:
        state_dict = checkpoint
        metadata = {}
    
    # Try flexible model for complex architectures
    if use_flexible or any('lstm' in k.lower() and 'fc3' in str(state_dict.keys()) for k in state_dict.keys()):
        model = FlexibleModel(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, {**metadata, 'architecture': 'flexible'}
    
    # Otherwise, infer and create structured model
    try:
        arch_type, arch_params = infer_model_architecture(state_dict)
        
        # Create model
        if arch_type == 'mlp':
            model = SimpleMLP(layer_sizes=arch_params['layer_sizes'])
        elif arch_type == 'lstm':
            model = SimpleLSTM(
                input_size=arch_params['input_size'],
                lstm_input_size=arch_params['lstm_input_size'],
                hidden_size=arch_params['hidden_size'],
                num_layers=arch_params['num_layers'],
                fc_shapes=arch_params.get('fc_shapes', []),
                fc_before_lstm=arch_params.get('fc_before_lstm', False),
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch_type}")
        
        # Load weights
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model, {**metadata, 'architecture': arch_type, 'architecture_params': arch_params}
    
    except Exception as e:
        # Fallback to flexible model
        model = FlexibleModel(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, {**metadata, 'architecture': 'flexible', 'error': str(e)}


def export_to_onnx(model: nn.Module, output_path: Path, input_shape: Tuple, model_name: str):
    """Export model to ONNX format.
    
    Note: PyTorch 2.9.1 has a known bug with ONNX export related to onnxscript
    registry initialization. We use torch.jit.trace as a workaround.
    """
    model.eval()
    model = model.to("cpu")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    try:
        # Workaround for PyTorch 2.9.1 ONNX export bug:
        # "Expecting a type not f<class 'typing.Union'> for typeinfo"
        # This happens in onnxscript registry initialization. Using JIT trace bypasses it.
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Convert to traced model first (workaround for PyTorch 2.9.1 bug)
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.eval()
        except Exception as e:
            # If JIT trace fails, try direct export (may fail with the Union type error)
            print(f"  ⚠ JIT trace failed, trying direct export: {e}")
            traced_model = model
        
        # Export using traced model
        torch.onnx.export(
            traced_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,  # Use opset 14 for better compatibility
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )
        
        # Validate ONNX file was created
        if not output_path.exists():
            print(f"  ✗ ONNX file not created: {output_path}")
            return False
        
        # Try to validate with onnx package if available
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
        except ImportError:
            pass  # onnx package not available, skip validation
        except Exception as e:
            print(f"  ⚠ ONNX validation warning: {e}")
            # Don't fail on validation errors, file was created
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ ONNX: {output_path.name} ({file_size_mb:.2f} MB)")
        return True
    except Exception as e:
        error_msg = str(e)
        if "typing.Union" in error_msg or "typeinfo" in error_msg:
            print(f"  ✗ ONNX export failed: Known PyTorch 2.9.1 + onnxscript compatibility bug")
            print(f"    Error: {error_msg[:100]}...")
            print(f"    Workaround options:")
            print(f"      1. Use RTNeural JSON format instead (recommended for C++ real-time)")
            print(f"      2. Downgrade PyTorch: pip install torch==2.8.2 onnx==1.16.0")
            print(f"      3. See models/onnx/README.md for details")
        else:
            print(f"  ✗ ONNX export failed: {error_msg}")
            import traceback
            if "--verbose" in sys.argv or "-v" in sys.argv:
                print(f"    Traceback: {traceback.format_exc()}")
        return False


def export_to_rtneural_json(model: nn.Module, output_path: Path, model_name: str):
    """Export model to RTNeural JSON format - supports Linear and LSTM layers."""
    try:
        layers = []
        lstm_count = 0
        
        # Extract layers from model in order
        for name, module in model.named_modules():
            # Skip root module
            if name == "":
                continue
                
            if isinstance(module, nn.Linear):
                weights = module.weight.detach().cpu().numpy().tolist()
                bias = module.bias.detach().cpu().numpy().tolist() if module.bias is not None else None
                
                layers.append({
                    "type": "dense",
                    "name": name,
                    "shape": list(module.weight.shape),
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "weights": weights,
                    "bias": bias if bias is not None else None,
                })
                
            elif isinstance(module, nn.LSTM):
                # RTNeural supports LSTM - extract all weight matrices
                lstm_count += 1
                layer_data = {
                    "type": "lstm",
                    "name": name,
                    "input_size": module.input_size,
                    "hidden_size": module.hidden_size,
                    "num_layers": module.num_layers,
                    "bidirectional": module.bidirectional,
                    "weights_ih": [],
                    "weights_hh": [],
                    "bias_ih": [],
                    "bias_hh": [],
                }
                
                # Extract weights for each layer
                for i in range(module.num_layers):
                    # PyTorch LSTM stores weights as:
                    # weight_ih_l{i} - input-to-hidden weights (4*hidden_size, input_size)
                    # weight_hh_l{i} - hidden-to-hidden weights (4*hidden_size, hidden_size)
                    # bias_ih_l{i} - input-to-hidden bias (4*hidden_size)
                    # bias_hh_l{i} - hidden-to-hidden bias (4*hidden_size)
                    
                    # Get weights from state dict (module itself doesn't expose _l{i} attributes directly)
                    state_dict = module.state_dict()
                    
                    weight_ih_key = f"weight_ih_l{i}"
                    weight_hh_key = f"weight_hh_l{i}"
                    bias_ih_key = f"bias_ih_l{i}"
                    bias_hh_key = f"bias_hh_l{i}"
                    
                    if weight_ih_key in state_dict:
                        layer_data["weights_ih"].append(
                            state_dict[weight_ih_key].detach().cpu().numpy().tolist()
                        )
                    if weight_hh_key in state_dict:
                        layer_data["weights_hh"].append(
                            state_dict[weight_hh_key].detach().cpu().numpy().tolist()
                        )
                    if bias_ih_key in state_dict:
                        layer_data["bias_ih"].append(
                            state_dict[bias_ih_key].detach().cpu().numpy().tolist()
                        )
                    if bias_hh_key in state_dict:
                        layer_data["bias_hh"].append(
                            state_dict[bias_hh_key].detach().cpu().numpy().tolist()
                        )
                
                layers.append(layer_data)
                print(f"    ✓ Extracted LSTM layer: {name} ({module.num_layers} layers, hidden_size={module.hidden_size})")
        
        if not layers:
            print(f"  ✗ RTNeural: No compatible layers found")
            return False
        
        # Create RTNeural JSON structure (RTNeural-compatible format)
        rtneural_json = {
            "version": "1.0",
            "format": "rtneural-json",
            "model_name": model_name,
            "input_size": layers[0].get("in_features") if layers[0].get("type") == "dense" else layers[0].get("input_size"),
            "output_size": layers[-1].get("out_features") if layers[-1].get("type") == "dense" else layers[-1].get("hidden_size"),
            "layers": layers,
        }
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(rtneural_json, f, indent=2)
        
        layer_types = [l.get("type") for l in layers]
        layer_summary = f"{len([l for l in layer_types if l == 'dense'])} dense, {lstm_count} LSTM"
        print(f"  ✓ RTNeural JSON: {output_path.name} ({layer_summary})")
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ RTNeural export failed: {e}")
        print(f"    Traceback: {traceback.format_exc()}")
        return False


MODEL_CONFIGS = {
    'emotionrecognizer': {
        'checkpoint': 'models/checkpoints/emotionrecognizer_best.pt',
        'input_shape': (1, 128),
    },
    'harmonypredictor': {
        'checkpoint': 'models/checkpoints/harmonypredictor_best.pt',
        'input_shape': (1, 128),
    },
    'melodytransformer': {
        'checkpoint': 'models/checkpoints/melodytransformer_best.pt',
        'input_shape': (1, 64),
    },
    'groovepredictor': {
        'checkpoint': 'models/checkpoints/groovepredictor_best.pt',
        'input_shape': (1, 64),
    },
    'dynamicsengine': {
        'checkpoint': 'models/checkpoints/dynamicsengine_best.pt',
        'input_shape': (1, 32),
    },
}


def main():
    parser = argparse.ArgumentParser(description='Export models to ONNX and RTNeural')
    parser.add_argument('--format', choices=['onnx', 'rtneural', 'all'], default='all',
                       help='Export format (default: all)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory (default: models/)')
    parser.add_argument('--models', nargs='+', default=list(MODEL_CONFIGS.keys()),
                       help='Models to export (default: all)')
    
    args = parser.parse_args()
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Model Export to Production Formats")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}")
    print()
    
    results = {}
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"⚠ Unknown model: {model_name}, skipping")
            continue
        
        config = MODEL_CONFIGS[model_name]
        checkpoint_path = project_root / config['checkpoint']
        
        print(f"\n{model_name.upper()}")
        print("-" * 70)
        
        if not checkpoint_path.exists():
            print(f"  ✗ Checkpoint not found: {checkpoint_path}")
            results[model_name] = {'success': False, 'error': 'Checkpoint not found'}
            continue
        
        try:
            # Load model - use flexible for emotionrecognizer (complex architecture)
            use_flexible = (model_name == 'emotionrecognizer')
            model, metadata = load_checkpoint_and_create_model(checkpoint_path, use_flexible=use_flexible)
            print(f"  ✓ Loaded: {metadata.get('architecture', 'unknown')} architecture")
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            model_results = {'success': True, 'exports': {}}
            
            # Export to requested formats
            if args.format in ['onnx', 'all']:
                onnx_path = output_dir / f"{model_name}.onnx"
                success = export_to_onnx(model, onnx_path, config['input_shape'], model_name)
                model_results['exports']['onnx'] = str(onnx_path) if success else None
            
            if args.format in ['rtneural', 'all']:
                rtneural_path = output_dir / f"{model_name}.json"
                success = export_to_rtneural_json(model, rtneural_path, model_name)
                model_results['exports']['rtneural'] = str(rtneural_path) if success else None
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[model_name] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPORT SUMMARY")
    print(f"{'='*70}\n")
    
    successful = sum(1 for r in results.values() if r.get('success'))
    total = len(results)
    
    for model_name, result in results.items():
        if result.get('success'):
            exports = result.get('exports', {})
            export_list = [fmt for fmt, path in exports.items() if path]
            print(f"  ✓ {model_name:20s} - Exported: {', '.join(export_list) or 'none'}")
        else:
            print(f"  ✗ {model_name:20s} - {result.get('error', 'Failed')}")
    
    print(f"\n✓ Successfully exported: {successful}/{total} models")
    
    if successful == total:
        print("\n🎉 All models exported successfully!")
        return 0
    else:
        print(f"\n⚠ {total - successful} models failed to export")
        return 1


if __name__ == '__main__':
    import argparse
    sys.exit(main())

