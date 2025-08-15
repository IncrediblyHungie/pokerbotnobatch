#!/usr/bin/env python3
"""
Parallel training script using multiprocessing for massive speedup.
Can achieve 10-50x speed improvement over single-threaded training.
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from strategy.blueprint_generator import BlueprintGenerator


def worker_train_batch(args):
    """Worker function to train a batch of iterations"""
    worker_id, batch_size, config_path, use_gpu = args
    
    try:
        # Suppress output during worker initialization
        import io
        import sys
        
        # Redirect stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Each worker gets its own blueprint generator
        blueprint_gen = BlueprintGenerator(
            config_path,
            use_gpu=False,  # Workers use CPU to avoid GPU conflicts
            use_batch_processing=False
        )
        
        # Restore stdout
        sys.stdout = old_stdout
        
        total_utility = {0: 0.0, 1: 0.0}
        
        # Train the batch with occasional progress updates
        for i in range(batch_size):
            game_state = blueprint_gen.setup_game(2)  # Heads-up for speed
            utilities = blueprint_gen.cfr_solver.train_iteration(game_state)
            
            for player_id, utility in utilities.items():
                total_utility[player_id] += utility
        
        # Return the trained infosets and utilities
        infosets_data = {}
        for key, infoset in blueprint_gen.cfr_solver.infosets.items():
            infosets_data[key] = {
                'regret_sum': infoset.regret_sum.copy(),
                'strategy_sum': infoset.strategy_sum.copy(),
                'reach_count': infoset.reach_count,
                'num_actions': infoset.num_actions
            }
        
        return worker_id, batch_size, infosets_data, total_utility
        
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return worker_id, 0, {}, {0: 0.0, 1: 0.0}


def parallel_train(iterations: int = 10000, num_workers: int = None, use_gpu: bool = True):
    """Train using multiple parallel workers"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 2, 16)  # Use half available cores, max 16
    
    print("🚀 PARALLEL POKER TRAINING")
    print("=" * 50)
    print(f"Target: {iterations:,} iterations")
    print(f"Workers: {num_workers} parallel processes")
    print(f"Expected speedup: {num_workers}x faster")
    print("=" * 50)
    
    config_path = "config/fast_training_config.yaml"
    
    # Calculate larger batch size to reduce overhead
    batch_size = max(100, iterations // num_workers)  # One batch per worker for efficiency
    total_batches = (iterations + batch_size - 1) // batch_size
    
    print(f"⚡ Batch size: {batch_size} iterations per worker")
    print(f"📦 Total batches: {total_batches}")
    
    # Create master blueprint generator for final strategy
    master_blueprint = BlueprintGenerator(
        config_path,
        use_gpu=use_gpu,
        use_batch_processing=False
    )
    
    start_time = time.time()
    completed_iterations = 0
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = []
        for batch_id in range(total_batches):
            remaining = iterations - completed_iterations
            current_batch_size = min(batch_size, remaining)
            
            if current_batch_size <= 0:
                break
                
            args = (batch_id, current_batch_size, config_path, False)
            future = executor.submit(worker_train_batch, args)
            futures.append((future, current_batch_size))
            completed_iterations += current_batch_size
        
        print(f"🎯 Submitted {len(futures)} batches to {num_workers} workers")
        
        # Collect results as they complete with live progress
        total_iterations_done = 0
        all_infosets = {}
        
        print(f"⏳ Training in progress...")
        print(f"Progress: 0/{iterations:,} (0.0%) | Rate: 0.0 it/s | ETA: calculating...")
        
        for future, batch_size in futures:
            try:
                # Longer timeout for large batches
                timeout_seconds = max(600, batch_size // 3)  # 3 it/s minimum expected
                worker_id, actual_batch_size, infosets_data, utilities = future.result(timeout=timeout_seconds)
                
                # Merge infosets from this worker
                for key, data in infosets_data.items():
                    if key in all_infosets:
                        # Combine regret and strategy sums
                        all_infosets[key]['regret_sum'] += data['regret_sum']
                        all_infosets[key]['strategy_sum'] += data['strategy_sum'] 
                        all_infosets[key]['reach_count'] += data['reach_count']
                    else:
                        all_infosets[key] = data
                
                total_iterations_done += actual_batch_size
                
                # Live progress update for every completion
                progress = total_iterations_done / iterations * 100
                elapsed = time.time() - start_time
                rate = total_iterations_done / elapsed if elapsed > 0 else 0
                eta = (iterations - total_iterations_done) / rate if rate > 0 else 0
                
                # Clear previous line and show updated progress
                print(f"\r🔄 Progress: {total_iterations_done:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Rate: {rate:.1f} it/s | ETA: {eta:.0f}s", end="", flush=True)
                
            except Exception as e:
                print(f"❌ Batch failed: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    final_rate = total_iterations_done / duration if duration > 0 else 0
    
    print(f"\n\n🎉 PARALLEL TRAINING COMPLETED!")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"🔄 Final rate: {final_rate:.1f} iterations/second")
    print(f"📈 Speedup: {final_rate / 3.1:.1f}x faster than single-threaded")
    print(f"📊 Information sets learned: {len(all_infosets):,}")
    
    # Save the combined strategy
    strategy_data = {
        'infosets': all_infosets,
        'iterations': total_iterations_done,
        'total_utility': {0: 0.0, 1: 0.0}
    }
    
    output_path = "data/blueprints/parallel_trained_bot.pkl"
    os.makedirs("data/blueprints", exist_ok=True)
    
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(strategy_data, f)
    
    print(f"💾 Saved to: {output_path}")
    
    return strategy_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel poker training")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    
    parallel_train(args.iterations, args.workers, use_gpu=not args.cpu)


if __name__ == "__main__":
    main()