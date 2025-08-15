"""
GameState class for tracking all poker game information.
Supports dynamic player count (2-10 players) and handles Texas Hold'em rules.
"""

from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np
from copy import deepcopy


class Action(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


class BettingRound(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Player:
    def __init__(self, player_id: int, stack: int):
        self.player_id = player_id
        self.stack = stack
        self.hole_cards = []
        self.is_active = True
        self.is_all_in = False
        self.bet_this_round = 0
        self.total_bet = 0
        self.folded = False

    def reset_for_new_round(self):
        """Reset player state for new betting round"""
        self.bet_this_round = 0

    def can_bet(self) -> bool:
        """Check if player can make a bet"""
        return self.is_active and not self.folded and not self.is_all_in and self.stack > 0


class GameState:
    def __init__(self, num_players: int, starting_stack: int, small_blind: int, big_blind: int):
        if num_players < 2 or num_players > 10:
            raise ValueError("Number of players must be between 2 and 10")
        
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Initialize players
        self.players = [Player(i, starting_stack) for i in range(num_players)]
        
        # Game state
        self.dealer_position = 0
        self.current_player = 0
        self.betting_round = BettingRound.PREFLOP
        self.community_cards = []
        self.pot = 0
        self.side_pots = []
        self.current_bet = 0
        self.min_raise = big_blind
        self.last_aggressive_player = None
        
        # Betting history for information sets
        self.betting_history = []
        self.round_betting_history = {
            BettingRound.PREFLOP: [],
            BettingRound.FLOP: [],
            BettingRound.TURN: [],
            BettingRound.RIVER: []
        }
        
        # Game flow control
        self.terminal = False
        self.winner = None
        
        # Initialize deck for dealing community cards
        self._deck = []
    
    def _create_and_shuffle_deck(self):
        """Create and shuffle a deck of cards"""
        import random
        from engine.hand_evaluator import Card, Rank, Suit
        
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        
        random.shuffle(deck)
        self._deck = deck
        
    def get_active_players(self) -> List[Player]:
        """Get list of players still in the hand"""
        return [p for p in self.players if p.is_active and not p.folded]
    
    def get_legal_actions(self, player_id: int) -> List[Tuple[Action, int]]:
        """Get legal actions for current player with bet amounts"""
        if self.terminal or player_id != self.current_player:
            return []
        
        player = self.players[player_id]
        if not player.can_bet():
            return []
        
        legal_actions = []
        
        # Fold is always legal (except when can check)
        if self.current_bet > player.bet_this_round:
            legal_actions.append((Action.FOLD, 0))
        
        # Check/Call
        if self.current_bet == player.bet_this_round:
            legal_actions.append((Action.CHECK, 0))
        else:
            call_amount = min(self.current_bet - player.bet_this_round, player.stack)
            if call_amount > 0:
                legal_actions.append((Action.CALL, call_amount))
        
        # Raise/Bet
        if player.stack > 0:
            call_amount = self.current_bet - player.bet_this_round
            remaining_stack = player.stack - call_amount
            
            if remaining_stack > 0:
                # Minimum raise
                min_raise_amount = max(self.min_raise, self.big_blind)
                if remaining_stack >= min_raise_amount:
                    legal_actions.append((Action.RAISE, call_amount + min_raise_amount))
                
                # All-in is always available if we have chips
                legal_actions.append((Action.ALL_IN, player.stack))
        
        return legal_actions
    
    def apply_action(self, player_id: int, action: Action, amount: int = 0) -> bool:
        """Apply an action and update game state"""
        if self.terminal or player_id != self.current_player:
            return False
        
        player = self.players[player_id]
        if not player.can_bet():
            return False
        
        # Record action in betting history
        action_record = (player_id, action, amount)
        self.betting_history.append(action_record)
        self.round_betting_history[self.betting_round].append(action_record)
        
        if action == Action.FOLD:
            player.folded = True
            player.is_active = False
        
        elif action == Action.CHECK:
            if self.current_bet != player.bet_this_round:
                return False  # Can't check if there's a bet to call
        
        elif action == Action.CALL:
            call_amount = min(self.current_bet - player.bet_this_round, player.stack)
            if amount != call_amount:
                return False
            self._place_bet(player, call_amount)
        
        elif action == Action.RAISE:
            call_amount = self.current_bet - player.bet_this_round
            total_bet = call_amount + (amount - call_amount)
            if total_bet > player.stack:
                return False
            self._place_bet(player, total_bet)
            self.current_bet = player.bet_this_round
            self.min_raise = amount - call_amount
            self.last_aggressive_player = player_id
        
        elif action == Action.ALL_IN:
            if amount != player.stack:
                return False
            self._place_bet(player, player.stack)
            player.is_all_in = True
            
            # Update current bet if this is a raise
            if player.bet_this_round > self.current_bet:
                raise_amount = player.bet_this_round - self.current_bet
                self.current_bet = player.bet_this_round
                self.min_raise = max(raise_amount, self.big_blind)
                self.last_aggressive_player = player_id
        
        # Move to next player
        self._advance_to_next_player()
        
        # Check if betting round is complete
        if self._is_betting_round_complete():
            self._advance_betting_round()
        
        return True
    
    def _place_bet(self, player: Player, amount: int):
        """Place a bet for a player"""
        if amount > player.stack:
            amount = player.stack
        
        player.stack -= amount
        player.bet_this_round += amount
        player.total_bet += amount
        self.pot += amount
        
        if player.stack == 0:
            player.is_all_in = True
    
    def _advance_to_next_player(self):
        """Move to the next active player"""
        original_player = self.current_player
        players_checked = 0
        
        while players_checked < self.num_players:
            self.current_player = (self.current_player + 1) % self.num_players
            players_checked += 1
            
            # If we've gone full circle, betting round might be complete
            if self.current_player == original_player:
                break
            
            player = self.players[self.current_player]
            if player.can_bet():
                break
    
    def _is_betting_round_complete(self) -> bool:
        """Check if current betting round is complete"""
        active_players = self.get_active_players()
        
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have equal bets or are all-in
        betting_players = [p for p in active_players if not p.is_all_in]
        
        if len(betting_players) == 0:
            return True
        
        if len(betting_players) == 1:
            # Only one player can bet, check if they've acted
            return self.current_player != betting_players[0].player_id or \
                   betting_players[0].bet_this_round == self.current_bet
        
        # All betting players must have equal bets
        bet_amounts = [p.bet_this_round for p in betting_players]
        return len(set(bet_amounts)) == 1 and all(amount == self.current_bet for amount in bet_amounts)
    
    def _advance_betting_round(self):
        """Advance to next betting round or end hand"""
        # Reset player bets for next round
        for player in self.players:
            player.reset_for_new_round()
        
        self.current_bet = 0
        self.min_raise = self.big_blind
        
        if self.betting_round == BettingRound.PREFLOP:
            self.betting_round = BettingRound.FLOP
            # Deal the flop (3 cards)
            if not hasattr(self, '_deck') or not self._deck:
                self._create_and_shuffle_deck()
            self.community_cards = self._deck[:3]
            self._deck = self._deck[3:]
        elif self.betting_round == BettingRound.FLOP:
            self.betting_round = BettingRound.TURN
            # Deal the turn (1 more card)
            if self._deck:
                self.community_cards.append(self._deck[0])
                self._deck = self._deck[1:]
        elif self.betting_round == BettingRound.TURN:
            self.betting_round = BettingRound.RIVER
            # Deal the river (1 more card)
            if self._deck:
                self.community_cards.append(self._deck[0])
                self._deck = self._deck[1:]
        else:
            # River complete - go to showdown
            self._end_hand()
            return
        
        # Reset current player to first active player for new betting round
        # In post-flop play, action starts with first active player after dealer
        self.current_player = (self.dealer_position + 1) % self.num_players
        
        # Find first active player from this position
        players_checked = 0
        while players_checked < self.num_players and not self.players[self.current_player].can_bet():
            self.current_player = (self.current_player + 1) % self.num_players
            players_checked += 1
        
        # If no player can bet, end the hand
        if players_checked >= self.num_players:
            self._end_hand()
    
    def _end_hand(self):
        """End the current hand"""
        self.terminal = True
        
        # Handle side pots if necessary
        active_players = self.get_active_players()
        if len(active_players) == 1:
            self.winner = active_players[0].player_id
            self._distribute_pot()
        else:
            # Multiple players - need showdown
            self._determine_winner_at_showdown(active_players)
    
    def _determine_winner_at_showdown(self, active_players: List[Player]):
        """Determine winner(s) at showdown using hand evaluation"""
        from engine.hand_evaluator import HandEvaluator
        
        evaluator = HandEvaluator()
        player_hands = []
        
        # Evaluate each player's hand (no debug output during training)
        for player in active_players:
            all_cards = player.hole_cards + self.community_cards
            if len(all_cards) >= 5:
                hand_rank, kickers = evaluator.evaluate_hand(all_cards)
                player_hands.append((player.player_id, hand_rank, kickers))
            else:
                # Should not happen in proper game flow
                player_hands.append((player.player_id, None, []))
        
        # Find the best hand(s)
        best_players = []
        best_hand_rank = None
        best_kickers = None
        
        for player_id, hand_rank, kickers in player_hands:
            if hand_rank is None:
                continue
                
            if best_hand_rank is None or hand_rank.value > best_hand_rank.value:
                best_players = [player_id]
                best_hand_rank = hand_rank
                best_kickers = kickers
            elif hand_rank.value == best_hand_rank.value:
                # Compare kickers
                kicker_comparison = self._compare_kickers(kickers, best_kickers)
                if kicker_comparison > 0:
                    best_players = [player_id]
                    best_kickers = kickers
                elif kicker_comparison == 0:
                    best_players.append(player_id)
        
        # Set winners (can be multiple for ties)
        if len(best_players) == 1:
            self.winner = best_players[0]
        else:
            self.winner = best_players  # Multiple winners (tie)
        
        # Actually distribute the pot to winner(s)
        self._distribute_pot()
    
    def _distribute_pot(self):
        """Distribute the pot to the winner(s)"""
        if self.winner is None or self.pot == 0:
            return
            
        if isinstance(self.winner, list):
            # Multiple winners - split pot
            pot_share = self.pot // len(self.winner)
            remainder = self.pot % len(self.winner)
            
            for i, winner_id in enumerate(self.winner):
                self.players[winner_id].stack += pot_share
                # Give remainder to first winner(s)
                if i < remainder:
                    self.players[winner_id].stack += 1
            
            pass  # Pot split silently
        else:
            # Single winner takes all
            self.players[self.winner].stack += self.pot
        
        # Reset pot
        self.pot = 0
    
    def _compare_kickers(self, kickers1: List[int], kickers2: List[int]) -> int:
        """Compare kickers. Returns 1 if kickers1 > kickers2, -1 if kickers1 < kickers2, 0 if equal"""
        for k1, k2 in zip(kickers1, kickers2):
            if k1 > k2:
                return 1
            elif k1 < k2:
                return -1
        return 0
    
    def is_terminal(self) -> bool:
        """Check if game state is terminal"""
        if self.terminal:
            return True
        
        active_players = self.get_active_players()
        return len(active_players) <= 1
    
    def get_infoset_key(self, player_id: int) -> str:
        """Generate information set key for a player"""
        player = self.players[player_id]
        
        # Hole cards (would be abstracted in real implementation)
        hole_cards_str = ",".join([str(card) for card in player.hole_cards])
        
        # Community cards (would be abstracted)
        community_str = ",".join([str(card) for card in self.community_cards])
        
        # Betting history for current round
        betting_str = "|".join([f"{pid}:{act.value}:{amt}" 
                               for pid, act, amt in self.round_betting_history[self.betting_round]])
        
        return f"{hole_cards_str}#{community_str}#{betting_str}#{self.betting_round.value}"
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state"""
        return deepcopy(self)
    
    def get_payoffs(self) -> List[float]:
        """Get payoffs for all players (used for CFR)"""
        payoffs = [0.0] * self.num_players
        
        if not self.terminal:
            return payoffs
        
        # Handle winner(s) - can be single winner or multiple winners (tie)
        if self.winner is not None:
            if isinstance(self.winner, list):
                # Multiple winners - split pot
                pot_share = float(self.pot) / len(self.winner)
                for winner_id in self.winner:
                    payoffs[winner_id] = pot_share
            else:
                # Single winner takes all
                payoffs[self.winner] = float(self.pot)
        
        # Subtract total bets from all players
        for i, player in enumerate(self.players):
            payoffs[i] -= float(player.total_bet)
        
        return payoffs