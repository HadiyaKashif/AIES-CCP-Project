import time
from flask import session, flash, redirect, url_for
from typing import List, Dict, Any, Optional
from components.utils import gemini_llm, get_session_value, set_session_value, flash_message, init_session_vars

def init_flashcard_session() -> None:
    """Initialize flashcard-related session variables."""
    init_session_vars({
        "flashcards": [],
        "flash_index": 0,
        "wrong_flashcards": [],
        "score": 0,
        "start_time": time.time(),
        "challenge_mode": False,
        "show_answer": False
    })

def generate_flashcards_from_text(text: str, num_cards: int = 5) -> List[Dict[str, str]]:
    """Generate flashcards from the given text using Gemini LLM."""
    prompt = f"""
    Generate {num_cards} flashcards (Question and Answer format) from the content below.
    Format:
    Q1: ...
    A1: ...
    Content:
    {text}
    """

    result = gemini_llm(prompt)
    cards = []
    lines = result.splitlines()
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            try:
                q = lines[i].split(":", 1)[1].strip()
                a = lines[i + 1].split(":", 1)[1].strip()
                cards.append({"question": q, "answer": a})
            except IndexError:
                continue
    
    return cards

def get_current_flashcard() -> Optional[Dict[str, str]]:
    """Get the current flashcard based on the session state."""
    cards = get_session_value('flashcards', [])
    idx = get_session_value('flash_index', 0)
    
    if not cards or idx >= len(cards):
        return None
    
    return cards[idx]

def get_flashcard_progress() -> Dict[str, Any]:
    """Get the current progress of the flashcard game."""
    cards = get_session_value('flashcards', [])
    idx = get_session_value('flash_index', 0)
    total = len(cards)
    score = get_session_value('score', 0)
    
    if not cards:
        return {
            'has_cards': False,
            'current_card': None,
            'progress': 0,
            'score': 0,
            'total': 0,
            'current': 0
        }
    
    return {
        'has_cards': True,
        'current_card': get_current_flashcard(),
        'progress': ((idx + 1) / total * 100) if total > 0 else 0,
        'score': score,
        'total': total,
        'current': idx + 1
    }

def handle_next_flashcard(action: str) -> None:
    """Handle the transition to the next flashcard."""
    if action == 'knew':
        set_session_value('score', get_session_value('score', 0) + 1)
    elif action == 'didnt_know':
        current_card = get_current_flashcard()
        if current_card:
            wrong_cards = get_session_value('wrong_flashcards', [])
            wrong_cards.append(current_card)
            set_session_value('wrong_flashcards', wrong_cards)
    
    # Update index and reset answer state
    new_index = get_session_value('flash_index', 0) + 1
    set_session_value('flash_index', new_index)
    set_session_value('show_answer', False)
    set_session_value('start_time', time.time())
    
    # Check if we're done with all cards
    cards = get_session_value('flashcards', [])
    if new_index >= len(cards):
        wrong_cards = get_session_value('wrong_flashcards', [])
        if wrong_cards:
            # Start reviewing wrong cards
            set_session_value('flashcards', wrong_cards)
            set_session_value('wrong_flashcards', [])
            set_session_value('flash_index', 0)
            flash_message('Repeating the cards you missed!', 'info')
        else:
            # Game complete
            flash_message(f'Game complete! Final score: {get_session_value("score", 0)}', 'success')
            set_session_value('flashcards', [])
            set_session_value('flash_index', 0)

def get_remaining_time() -> int:
    """Get remaining time for challenge mode."""
    if not get_session_value('challenge_mode', False):
        return -1
    
    start_time = get_session_value('start_time', time.time())
    elapsed = time.time() - start_time
    remaining = max(0, 15 - int(elapsed))
    return remaining

def toggle_challenge_mode() -> None:
    """Toggle the challenge mode state."""
    current_mode = get_session_value('challenge_mode', False)
    set_session_value('challenge_mode', not current_mode)
    set_session_value('start_time', time.time())

def toggle_answer() -> None:
    """Toggle the answer visibility state."""
    current_state = get_session_value('show_answer', False)
    set_session_value('show_answer', not current_state)
