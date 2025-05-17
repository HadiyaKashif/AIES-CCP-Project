import streamlit as st
import time

def flashcard_ui():
    # Initialize session state variables
    for key, default in {
        "flashcards": [],
        "flash_index": 0,
        "wrong_flashcards": [],
        "score": 0,
        "start_time": time.time(),
        "challenge_mode": False,
        "generate_flashcards": False,
        "show_answer": False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown("<h1 style='text-align:center; margin-bottom:1rem;'>🧠 Flashcard Game</h1>", unsafe_allow_html=True)

    # Challenge mode toggle
    with st.expander("⚙️ Settings"):
        st.session_state.challenge_mode = st.toggle("⏱️ Enable Challenge Mode (15s per card)", value=st.session_state.challenge_mode)

    st.markdown("---")

    # Generate flashcards button
    if st.button("📄 Generate Flashcards from Context"):
        st.session_state.generate_flashcards = True

    if st.session_state.generate_flashcards:
        if "vector_store" in st.session_state:
            docs = st.session_state.vector_store.similarity_search("summary", k=20)
            full_text = " ".join([doc.page_content for doc in docs])
            st.session_state.flashcards = generate_flashcards_from_text(full_text)
            st.session_state.flash_index = 0
            st.session_state.wrong_flashcards = []
            st.session_state.score = 0
            st.session_state.start_time = time.time()
            st.session_state.show_answer = False
            st.session_state.generate_flashcards = False
            st.success("✅ Flashcards created!")
            st.rerun()
        else:
            st.warning("❗ Please upload documents first.")
            st.session_state.generate_flashcards = False

    # Main flashcard display
    if st.session_state.flashcards:
        cards = st.session_state.flashcards
        idx = st.session_state.flash_index
        total = len(cards)

        # Handle end of flashcards
        if idx >= total:
            next_flashcard()
            return

        card = cards[idx]

        # Time tracking
        if st.session_state.challenge_mode:
            elapsed = time.time() - st.session_state.start_time
            remaining = max(0, 15 - int(elapsed))
            st.markdown(f"<p style='color:#d9534f'><strong>⏳ Time Left:</strong> {remaining}s</p>", unsafe_allow_html=True)
            if remaining == 0:
                st.warning("⏱️ Time's up!")
                next_flashcard()
                st.rerun()

        # Progress and Score
        st.progress((idx + 1) / total)
        st.markdown(f"📘 Flashcard {idx + 1}/{total} &nbsp;&nbsp;|&nbsp;&nbsp; 🏅 Score: {st.session_state.score}", unsafe_allow_html=True)

        # Flashcard UI container
        with st.container():
            html = f"""
            <div style='
                background-color: #f8f9fa;
                color: #212529;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                margin-top: 1rem;
                margin-bottom: 1rem;
            '>
                <h3 style='margin-bottom: 1rem;'>❓ {card['question']}</h3>
            """

        if st.session_state.show_answer:
            html += f"""
            <div style='color:#155724;background-color:#d4edda;padding:1rem;border-radius:10px;'>
                <strong>💡 Answer:</strong> {card['answer']}
            </div>
            """
        else:
            if st.button("🔍 Show Answer"):
                st.session_state.show_answer = True
                st.rerun()

        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ I knew this"):
                st.session_state.score += 1
                next_flashcard()
                st.rerun()
        with col2:
            if st.button("❌ I didn’t know"):
                st.session_state.wrong_flashcards.append(card)
                next_flashcard()
                st.rerun()

    else:
        st.info("Click the button above to generate flashcards.")


def next_flashcard():
    st.session_state.flash_index += 1
    st.session_state.start_time = time.time()
    st.session_state.show_answer = False

    # If done, repeat wrong ones or finish
    if st.session_state.flash_index >= len(st.session_state.flashcards):
        if st.session_state.wrong_flashcards:
            st.success("🔁 Repeating the flashcards you missed!")
            st.session_state.flashcards = st.session_state.wrong_flashcards
            st.session_state.wrong_flashcards = []
            st.session_state.flash_index = 0
            st.session_state.start_time = time.time()
            st.session_state.show_answer = False
        else:
            st.balloons()
            st.success("🎉 All flashcards reviewed!")
            st.toast(f"🏁 Final Score: {st.session_state.score}", icon="✅")
            if st.button("🔄 Restart Game"):
                st.session_state.flashcards = []
                st.session_state.flash_index = 0
                st.session_state.wrong_flashcards = []
                st.session_state.score = 0
                st.session_state.show_answer = False
                st.rerun()


def generate_flashcards_from_text(text, num_cards=5):
    from components.utils import gemini_llm

    prompt = f"""
    Generate {num_cards} flashcards (Question and Answer format) from the content below.
    Format:
    Q1: ...
    A1: ...
    Content:
    {text}
    """

    result = gemini_llm(prompt)
    st.write("🔍 Raw Flashcard Output:", result)  # Debug

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
