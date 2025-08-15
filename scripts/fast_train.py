#!/usr/bin/env python3
"""
Fast training script optimized for speed over accuracy.
Good for quick testing and prototyping.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator


def fast_train(iterations: int = 10000, use_gpu: bool = True):
    """Fast training with all optimizations enabled"""
    
    print("🚀 FAST POKER TRAINING")
    print("=" * 50)
    print(f"Target: {iterations:,} iterations")
    print(f"Mode: {'GPU' if use_gpu else 'CPU'} accelerated")
    print(f"Focus: Speed over accuracy")
    print("=" * 50)
    
    config_path = "config/fast_training_config.yaml"
    
    # Use fast training config with reliable SimpleCFR
    blueprint_gen = BlueprintGenerator(
        config_path,
        use_gpu=use_gpu,
        device_id=0,
        use_batch_processing=False,  # Disable problematic batching
        batch_size=64  # Not used with SimpleCFR
    )
    
    # Skip abstraction loading for pure speed
    print("⚡ Skipping abstraction loading for maximum speed")
    
    # Start training with timing
    start_time = time.time()
    
    try:
        stats = blueprint_gen.train_blueprint(
            iterations=iterations,
            checkpoint_frequency=max(1000, iterations // 10)  # Fewer checkpoints
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"🔄 Rate: {iterations / duration:.1f} iterations/second")
        print(f"📊 Information sets learned: {len(blueprint_gen.cfr_solver.infosets):,}")
        
        # Save final strategy
        final_path = "data/blueprints/fast_trained_bot.pkl"
        blueprint_gen.cfr_solver.save_strategy(final_path)
        print(f"💾 Saved to: {final_path}")
        
        return stats
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted")
        current_iterations = blueprint_gen.cfr_solver.iterations
        duration = time.time() - start_time
        print(f"⏱️  Completed {current_iterations:,} iterations in {duration:.1f}s")
        print(f"🔄 Rate: {current_iterations / duration:.1f} iterations/second")
        
        # Save partial progress
        partial_path = f"data/blueprints/partial_{current_iterations}.pkl"
        blueprint_gen.cfr_solver.save_strategy(partial_path)
        print(f"💾 Partial progress saved to: {partial_path}")


def benchmark_training_speed():
    """Benchmark different configurations to find optimal settings"""
    
    print("🏁 TRAINING SPEED BENCHMARK")
    print("=" * 50)
    
    configs = [
        ("CPU SimpleCFR", False, False, 64),
        ("GPU SimpleCFR", True, False, 64),
        ("GPU BatchCFR", True, True, 256),
        ("GPU BatchCFR Large", True, True, 512),
    ]
    
    test_iterations = 100
    
    for name, use_gpu, use_batch, batch_size in configs:
        print(f"\n📊 Testing: {name}")
        print("-" * 30)
        
        try:
            blueprint_gen = BlueprintGenerator(
                "config/fast_training_config.yaml",
                use_gpu=use_gpu,
                use_batch_processing=use_batch,
                batch_size=batch_size
            )
            
            start_time = time.time()
            
            # Run small test
            stats = blueprint_gen.train_blueprint(iterations=test_iterations)
            
            duration = time.time() - start_time
            rate = test_iterations / duration
            
            print(f"⏱️  {duration:.2f}s for {test_iterations} iterations")
            print(f"🔄 Rate: {rate:.1f} iterations/second")
            print(f"📈 Projected 10K rate: {10000 / rate:.0f} seconds")
            
        except Exception as e:
            print(f"❌ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fast poker bot training")
    parser.add_argument("--iterations", type=int, default=10000,
                       help="Number of training iterations")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU-only mode")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run training speed benchmark")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_training_speed()
    else:
        fast_train(args.iterations, use_gpu=not args.cpu)


if __name__ == "__main__":
    main()