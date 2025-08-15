"""
Simple card abstraction that actually works - bypasses the complex clustering.
"""

import numpy as np
from typing import List, Dict, Tuple
from engine.hand_evaluator import Card, Rank, Suit
from engine.game_state import BettingRound


class SimpleCardAbstraction:
    """Simple but functional card abstraction"""
    
    def __init__(self, config: dict, device_config=None):
        self.config = config
        
        # Initialize preflop mapping (deterministic)
        self.preflop_mapping = self._create_preflop_mapping()
        print(f"Created {len(self.preflop_mapping)} preflop buckets")
        
    def _create_preflop_mapping(self) -> Dict[str, int]:
        """Create simple preflop hand mapping"""
        mapping = {}
        bucket_id = 0
        
        # Generate all possible two-card combinations
        for rank1 in Rank:
            for rank2 in Rank:
                r1_val, r2_val = rank1.value, rank2.value
                
                # Ensure consistent ordering (higher rank first)
                if r1_val < r2_val:
                    r1_val, r2_val = r2_val, r1_val
                    rank1, rank2 = rank2, rank1
                
                if r1_val == r2_val:
                    # Pocket pair
                    key = f"{r1_val}{r1_val}"
                    if key not in mapping:
                        mapping[key] = bucket_id
                        bucket_id += 1
                else:
                    # Suited and offsuit
                    key_suited = f"{r1_val}{r2_val}s"
                    key_offsuit = f"{r1_val}{r2_val}o"
                    
                    if key_suited not in mapping:
                        mapping[key_suited] = bucket_id
                        bucket_id += 1
                    if key_offsuit not in mapping:
                        mapping[key_offsuit] = bucket_id
                        bucket_id += 1
        
        return mapping
    
    def _canonicalize_preflop_hand(self, hole_cards: List[Card]) -> str:
        """Convert preflop hand to canonical string"""
        card1, card2 = hole_cards[0], hole_cards[1]
        rank1, rank2 = card1.rank.value, card2.rank.value
        
        # Sort by rank (higher first)
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1
            card1, card2 = card2, card1
        
        # Determine if suited
        suited = card1.suit == card2.suit
        
        # Create canonical representation
        if rank1 == rank2:
            return f"{rank1}{rank1}"  # Pocket pair
        else:
            suffix = "s" if suited else "o"
            return f"{rank1}{rank2}{suffix}"
    
    def get_bucket(self, hole_cards: List[Card], board: List[Card], 
                   betting_round: BettingRound) -> int:
        """Get bucket number for given hand and board"""
        if betting_round == BettingRound.PREFLOP:
            key = self._canonicalize_preflop_hand(hole_cards)
            return self.preflop_mapping.get(key, 0)
        
        # Simple post-flop abstraction based on hand strength
        all_cards = hole_cards + board
        if len(all_cards) < 5:
            return 0
            
        # Very simple hand strength bucketing
        hand_strength = self._evaluate_simple_strength(all_cards)
        
        # Map to buckets
        if betting_round == BettingRound.FLOP:
            return int(hand_strength * 100) % 100
        elif betting_round == BettingRound.TURN:
            return int(hand_strength * 100) % 100
        elif betting_round == BettingRound.RIVER:
            return int(hand_strength * 100) % 100
        
        return 0
    
    def _evaluate_simple_strength(self, cards: List[Card]) -> float:
        """Simple hand strength evaluation (0.0 to 1.0)"""
        if len(cards) < 5:
            return 0.5
        
        # Count ranks and suits
        ranks = [card.rank.value for card in cards]
        suits = [card.suit for card in cards]
        
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        count_values = sorted(rank_counts.values(), reverse=True)
        has_flush = max(suit_counts.values()) >= 5
        
        # Simple hand type detection
        if count_values[0] == 4:
            return 0.95  # Four of a kind
        elif count_values[0] == 3 and count_values[1] == 2:
            return 0.90  # Full house
        elif has_flush:
            return 0.85  # Flush
        elif count_values[0] == 3:
            return 0.70  # Three of a kind
        elif count_values[0] == 2 and count_values[1] == 2:
            return 0.60  # Two pair
        elif count_values[0] == 2:
            return 0.50  # One pair
        else:
            # High card - based on highest card
            high_card = max(ranks)
            return 0.3 + (high_card - 2) / 39.0  # Scale 2-14 to 0.3-0.6
    
    def load_abstractions(self, path: str):
        """Compatibility method - no-op for simple abstraction"""
        pass
        
    def save_abstractions(self, path: str):
        """Compatibility method - no-op for simple abstraction"""
        pass