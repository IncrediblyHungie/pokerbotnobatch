#!/usr/bin/env python3
"""
Resumable parallel training script with checkpoint/resume capability.
Can survive restarts and continue training from where it left off.
"""

import os
import sys
import time
import pickle
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
        import yaml
        
        # Load config to get num_players
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        num_players = config['game']['num_players']
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        blueprint_gen = BlueprintGenerator(
            config_path,
            use_gpu=False,
            use_batch_processing=False
        )
        
        sys.stdout = old_stdout
        
        total_utility = {i: 0.0 for i in range(num_players)}
        
        # Train the batch
        for i in range(batch_size):
            game_state = blueprint_gen.setup_game(num_players)
            utilities = blueprint_gen.cfr_solver.train_iteration(game_state)
            
            for player_id, utility in utilities.items():
                if player_id in total_utility:
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
        return worker_id, 0, {}, {}


def save_checkpoint(checkpoint_path, iterations_completed, all_infosets, start_time):
    """Save training checkpoint"""
    checkpoint_data = {
        'iterations_completed': iterations_completed,
        'infosets': all_infosets,
        'start_time': start_time,
        'checkpoint_time': time.time(),
        'version': '1.0'
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"\n💾 Checkpoint saved: {iterations_completed:,} iterations")


def load_checkpoint(checkpoint_path):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"📁 Checkpoint found: {checkpoint_data['iterations_completed']:,} iterations")
        return checkpoint_data
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        return None


def resumable_parallel_train(total_iterations: int = 50000, num_workers: int = None, 
                            checkpoint_name: str = None, checkpoint_frequency: int = 5000):
    """Train with checkpoint/resume capability"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 2, 32)
    
    if checkpoint_name is None:
        checkpoint_name = f"training_{total_iterations}"
    
    checkpoint_path = f"data/checkpoints/{checkpoint_name}.pkl"
    
    print("🚀 RESUMABLE PARALLEL POKER TRAINING")
    print("=" * 50)
    print(f"Target: {total_iterations:,} iterations")
    print(f"Workers: {num_workers} parallel processes")
    print(f"Checkpoint: {checkpoint_name}")
    print("=" * 50)
    
    # Try to load existing checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    if checkpoint_data:
        iterations_completed = checkpoint_data['iterations_completed']
        all_infosets = checkpoint_data['infosets']
        original_start_time = checkpoint_data['start_time']
        
        print(f"🔄 Resuming from {iterations_completed:,}/{total_iterations:,} iterations")
        remaining_iterations = total_iterations - iterations_completed
        
        if remaining_iterations <= 0:
            print("✅ Training already completed!")
            return checkpoint_data
    else:
        print("🆕 Starting fresh training")
        iterations_completed = 0
        all_infosets = {}
        original_start_time = time.time()
        remaining_iterations = total_iterations
    
    # Use command line config or default
    import sys
    if len(sys.argv) > 1:
        if 'high_quality' in ' '.join(sys.argv):
            config_path = "config/high_quality_training.yaml"
        elif 'pluribus_6p' in ' '.join(sys.argv):
            config_path = "config/working_six_player.yaml"
        else:
            config_path = "config/working_training_config.yaml"
    else:
        config_path = "config/working_training_config.yaml"
    
    # Calculate batch size for remaining work
    batch_size = max(100, remaining_iterations // num_workers)
    total_batches = (remaining_iterations + batch_size - 1) // batch_size
    
    print(f"⚡ Remaining: {remaining_iterations:,} iterations")
    print(f"📦 Batch size: {batch_size} iterations per worker")
    print(f"🎯 Total batches: {total_batches}")
    
    current_start_time = time.time()
    
    # Process remaining batches
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Submit batches for remaining work
        iterations_to_submit = 0
        for batch_id in range(total_batches):
            current_batch_size = min(batch_size, remaining_iterations - iterations_to_submit)
            
            if current_batch_size <= 0:
                break
                
            args = (batch_id, current_batch_size, config_path, False)
            future = executor.submit(worker_train_batch, args)
            futures.append((future, current_batch_size))
            iterations_to_submit += current_batch_size
        
        print(f"🎯 Submitted {len(futures)} batches to {num_workers} workers")
        print(f"⏳ Training in progress...")
        
        batch_completions = 0
        
        for future, batch_size in futures:
            try:
                timeout_seconds = max(600, batch_size // 3)
                worker_id, actual_batch_size, infosets_data, utilities = future.result(timeout=timeout_seconds)
                
                # Merge infosets
                for key, data in infosets_data.items():
                    if key in all_infosets:
                        all_infosets[key]['regret_sum'] += data['regret_sum']
                        all_infosets[key]['strategy_sum'] += data['strategy_sum'] 
                        all_infosets[key]['reach_count'] += data['reach_count']
                    else:
                        all_infosets[key] = data
                
                iterations_completed += actual_batch_size
                batch_completions += 1
                
                # Live progress update
                progress = iterations_completed / total_iterations * 100
                session_elapsed = time.time() - current_start_time
                total_elapsed = time.time() - original_start_time
                session_rate = (iterations_completed - (total_iterations - remaining_iterations)) / session_elapsed if session_elapsed > 0 else 0
                overall_rate = iterations_completed / total_elapsed if total_elapsed > 0 else 0
                eta = (total_iterations - iterations_completed) / session_rate if session_rate > 0 else 0
                
                print(f"\r🔄 Progress: {iterations_completed:,}/{total_iterations:,} ({progress:.1f}%) | "
                      f"Rate: {session_rate:.1f} it/s | Overall: {overall_rate:.1f} it/s | ETA: {eta:.0f}s", 
                      end="", flush=True)
                
                # Save checkpoint periodically
                if iterations_completed % checkpoint_frequency == 0 or batch_completions % 5 == 0:
                    save_checkpoint(checkpoint_path, iterations_completed, all_infosets, original_start_time)
                
            except Exception as e:
                print(f"\n❌ Batch failed: {e}")
    
    # Final results
    total_duration = time.time() - original_start_time
    session_duration = time.time() - current_start_time
    final_rate = iterations_completed / total_duration if total_duration > 0 else 0
    
    print(f"\n\n🎉 TRAINING COMPLETED!")
    print(f"⏱️  Total duration: {total_duration:.1f} seconds")
    print(f"⏱️  Session duration: {session_duration:.1f} seconds")
    print(f"🔄 Final rate: {final_rate:.1f} iterations/second")
    print(f"📈 Speedup: {final_rate / 3.1:.1f}x faster than single-threaded")
    print(f"📊 Information sets learned: {len(all_infosets):,}")
    
    # Save final strategy
    strategy_data = {
        'infosets': all_infosets,
        'iterations': iterations_completed,
        'total_utility': {}
    }
    
    final_path = f"data/blueprints/{checkpoint_name}_final.pkl"
    os.makedirs("data/blueprints", exist_ok=True)
    
    with open(final_path, 'wb') as f:
        pickle.dump(strategy_data, f)
    
    print(f"💾 Final strategy saved to: {final_path}")
    
    # Save final checkpoint
    save_checkpoint(checkpoint_path, iterations_completed, all_infosets, original_start_time)
    
    return strategy_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Resumable parallel poker training")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", type=str, default=None, help="Checkpoint name")
    parser.add_argument("--checkpoint-freq", type=int, default=5000, help="Checkpoint frequency")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume specific checkpoint")
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        checkpoint_dir = "data/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
            if checkpoints:
                print("📁 Available checkpoints:")
                for cp in sorted(checkpoints):
                    cp_path = os.path.join(checkpoint_dir, cp)
                    try:
                        with open(cp_path, 'rb') as f:
                            data = pickle.load(f)
                        name = cp.replace('.pkl', '')
                        completed = data.get('iterations_completed', 0)
                        checkpoint_time = data.get('checkpoint_time', 0)
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_time))
                        print(f"  • {name}: {completed:,} iterations ({timestamp})")
                    except:
                        print(f"  • {cp}: (corrupted)")
            else:
                print("No checkpoints found")
        else:
            print("No checkpoint directory found")
        return
    
    checkpoint_name = args.resume or args.name
    if not checkpoint_name and args.resume:
        checkpoint_name = args.resume
    
    resumable_parallel_train(
        total_iterations=args.iterations,
        num_workers=args.workers,
        checkpoint_name=checkpoint_name,
        checkpoint_frequency=args.checkpoint_freq
    )


if __name__ == "__main__":
    main()