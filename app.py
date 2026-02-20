import streamlit as st
import pandas as pd
import re

# ── Configuration ──────────────────────────────────────────────────────────────
MY_NAME = "Iván Nicolás Gutiérrez Arias"

GAMES = ["Tango", "Queens", "Zip", "Mini Sudoku"]

GAME_ICONS = {
    "Tango":       "🟡",
    "Queens":      "👑",
    "Zip":         "⚡",
    "Mini Sudoku": "🔢",
}

# Matches all four game formats:
#   "Tango n.º 240 | 0:46"
#   "Queens n.º 400 | 1:32"
#   "Mini Sudoku n.º 193 | 1:41"
#   "Zip #78 | 0:13"
# Mini Sudoku must appear first in the alternation to avoid partial matches.
GAME_RE = re.compile(
    r"(Mini Sudoku|Tango|Queens|Zip)"
    r"\s+(?:n\.?[º°]|#)\s*(\d+)"
    r"\s*\|\s*(\d+):(\d+)",
    re.IGNORECASE,
)

_EMPTY_RESULTS = pd.DataFrame(
    columns=["sender", "date", "game", "puzzle_num", "time_sec"]
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def to_seconds(minutes: str, seconds: str) -> int:
    return int(minutes) * 60 + int(seconds)


def fmt_time(total_seconds: int) -> str:
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def parse_conversation(text: str, my_name: str, contact_name: str) -> tuple[list[dict], bool]:
    """
    Parse a full copy-pasted LinkedIn conversation.

    LinkedIn headers look like  "Full Name   HH:MM"  (name + 2+ spaces + time).
    We find every occurrence of either participant's name at the start of a line,
    build a sorted list of (position, sender) markers, then attribute each game
    result to whichever speaker marker precedes it in the text.

    Returns (records, names_were_detected).
    """
    def speaker_re(name: str) -> re.Pattern:
        # Matches a line that STARTS with the full name followed by any whitespace
        # (covers "Name   8:41" and "Name ha enviado los siguientes mensajes…")
        return re.compile(r"^" + re.escape(name) + r"\s", re.IGNORECASE | re.MULTILINE)

    speaker_markers: list[tuple[int, str]] = []
    for name in (my_name, contact_name):
        for m in speaker_re(name).finditer(text):
            speaker_markers.append((m.start(), name))

    speaker_markers.sort()
    names_detected = bool(speaker_markers)

    records = []
    for m in GAME_RE.finditer(text):
        pos  = m.start()
        game = next((g for g in GAMES if g.lower() == m.group(1).lower()), m.group(1))

        # Walk backwards through markers to find the last speaker before this result
        sender = None
        for sp_pos, sp_name in reversed(speaker_markers):
            if sp_pos < pos:
                sender = sp_name
                break

        if sender:
            records.append({
                "sender":     sender,
                "date":       None,
                "game":       game,
                "puzzle_num": int(m.group(2)),
                "time_sec":   to_seconds(m.group(3), m.group(4)),
            })

    return records, names_detected


def parse_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw LinkedIn messages CSV into a tidy game-results table."""
    col = {c.upper(): c for c in df.columns}
    from_col    = col.get("FROM",    None)
    content_col = col.get("CONTENT", None)
    date_col    = col.get("DATE",    None)

    records = []
    for _, row in df.iterrows():
        content  = str(row[content_col]) if content_col else ""
        sender   = str(row[from_col]).strip() if from_col else ""
        date_str = str(row[date_col]) if date_col else ""

        m = GAME_RE.search(content)
        if not m:
            continue

        game = next((g for g in GAMES if g.lower() == m.group(1).lower()), m.group(1))

        try:
            date = pd.to_datetime(date_str, utc=True).date()
        except Exception:
            date = None

        records.append({
            "sender":     sender,
            "date":       date,
            "game":       game,
            "puzzle_num": int(m.group(2)),
            "time_sec":   to_seconds(m.group(3), m.group(4)),
        })

    return pd.DataFrame(records) if records else _EMPTY_RESULTS.copy()


def merge_results(csv_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine CSV and manually-pasted results.
    For the same (sender, game, puzzle_num), keep the minimum (best) time.
    """
    if manual_df.empty:
        return csv_df
    combined = pd.concat([csv_df, manual_df], ignore_index=True)
    return (
        combined
        .sort_values("time_sec")
        .drop_duplicates(subset=["sender", "game", "puzzle_num"], keep="first")
        .reset_index(drop=True)
    )


def compute_scores(results: pd.DataFrame, my_name: str, contact: str) -> dict:
    """
    For each game, match shared puzzle numbers between me and the contact.
    Lower time wins. Returns a dict keyed by game name.
    """
    my_df = results[results["sender"] == my_name]
    co_df = results[results["sender"] == contact]

    scores = {g: {"me": 0, "contact": 0, "tie": 0, "duels": []} for g in GAMES}

    for game in GAMES:
        my_g = my_df[my_df["game"] == game].groupby("puzzle_num")["time_sec"].min()
        co_g = co_df[co_df["game"] == game].groupby("puzzle_num")["time_sec"].min()

        for pnum in sorted(my_g.index.intersection(co_g.index)):
            mt = int(my_g[pnum])
            ct = int(co_g[pnum])

            if mt < ct:
                winner = "me"
                scores[game]["me"] += 1
            elif ct < mt:
                winner = "contact"
                scores[game]["contact"] += 1
            else:
                winner = "tie"
                scores[game]["tie"] += 1

            scores[game]["duels"].append({
                "puzzle_num":   pnum,
                "my_time":      mt,
                "contact_time": ct,
                "winner":       winner,
            })

    return scores


# ── Streamlit App ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="LinkedIn Games Tracker",
        page_icon="🎮",
        layout="wide",
    )

    # ── Session state ──────────────────────────────────────────────────────────
    if "manual_results" not in st.session_state:
        st.session_state.manual_results = _EMPTY_RESULTS.copy()

    st.title("🎮 LinkedIn Games Tracker")
    st.caption("Compare your LinkedIn mini-game scores against your contacts.")

    # ── File upload ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload your LinkedIn `messages.csv`", type=["csv"])

    if uploaded is None:
        st.info(
            "⬆️ Upload a LinkedIn messages export to get started.\n\n"
            "**How to export:** LinkedIn → Settings → Data Privacy → "
            "Get a copy of your data → Messages"
        )
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        return

    missing_cols = {"FROM", "CONTENT", "DATE"} - {c.upper() for c in df.columns}
    if missing_cols:
        st.error(
            f"Expected columns not found: **{missing_cols}**\n\n"
            f"Columns in your file: `{list(df.columns)}`"
        )
        return

    csv_results = parse_messages(df)
    results     = merge_results(csv_results, st.session_state.manual_results)

    if results.empty:
        st.warning("No recognised game results were found.")
        return

    all_senders = results["sender"].unique().tolist()

    if MY_NAME not in all_senders:
        st.error(
            f"Your name **{MY_NAME}** was not found among message senders.\n\n"
            f"Detected senders: {all_senders}\n\n"
            "Edit the `MY_NAME` constant at the top of `app.py` if needed."
        )
        return

    contacts = sorted(s for s in all_senders if s != MY_NAME)

    if not contacts:
        st.warning("No other contacts with game results were found.")
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Overview metrics
        st.header("📊 Overview")
        manual_count = len(st.session_state.manual_results)
        st.metric("Game results (CSV)",    len(csv_results))
        st.metric("Manually added",        manual_count)
        st.metric("Total",                 len(results))
        st.metric("Contacts with results", len(contacts))
        st.markdown("**Results by game**")
        st.bar_chart(results["game"].value_counts())

        st.divider()

        # ── Paste recent messages ──────────────────────────────────────────────
        st.subheader("✍️ Add recent messages")
        st.caption(
            "Open a LinkedIn conversation, select all, copy, and paste below. "
            "Speaker attribution is detected automatically."
        )

        # Contact picker: existing contacts + free-text for new ones
        NEW_CONTACT = "— New contact —"
        paste_contact_pick = st.selectbox(
            "Contact", contacts + [NEW_CONTACT], key="paste_contact_pick"
        )
        if paste_contact_pick == NEW_CONTACT:
            paste_contact = st.text_input(
                "Enter contact name", key="new_contact_name"
            ).strip()
        else:
            paste_contact = paste_contact_pick

        convo_paste = st.text_area(
            "Full conversation",
            height=220,
            key="convo_paste",
            placeholder="Paste the full LinkedIn conversation here…",
        )

        btn_add, btn_clear = st.columns(2)

        if btn_add.button("Add", use_container_width=True, type="primary"):
            if not paste_contact:
                st.error("Select or enter a contact name first.")
            elif not convo_paste.strip():
                st.warning("Paste some conversation text first.")
            else:
                records, names_detected = parse_conversation(
                    convo_paste, MY_NAME, paste_contact
                )
                if not names_detected:
                    st.error(
                        f"Could not find **{MY_NAME}** or **{paste_contact}** "
                        "in the pasted text. Make sure the full names match exactly."
                    )
                elif not records:
                    st.warning("Names were found but no game results were detected.")
                else:
                    my_count = sum(1 for r in records if r["sender"] == MY_NAME)
                    co_count = len(records) - my_count
                    first    = paste_contact.split()[0]
                    new_df   = pd.DataFrame(records)
                    st.session_state.manual_results = pd.concat(
                        [st.session_state.manual_results, new_df],
                        ignore_index=True,
                    )
                    st.success(
                        f"Added {len(records)} result{'s' if len(records) > 1 else ''}: "
                        f"{my_count} for Iván, {co_count} for {first}."
                    )
                    st.rerun()

        if btn_clear.button("Clear all", use_container_width=True):
            st.session_state.manual_results = _EMPTY_RESULTS.copy()
            st.rerun()

    # ── Contact selector ───────────────────────────────────────────────────────
    contact = st.selectbox("🔍 Compare against:", contacts)
    if not contact:
        return

    scores        = compute_scores(results, MY_NAME, contact)
    contact_first = contact.split()[0]

    # ── Per-game scorecards ────────────────────────────────────────────────────
    st.markdown(f"## Iván vs {contact}")
    cols   = st.columns(len(GAMES))
    totals = {"me": 0, "contact": 0, "tie": 0}

    for i, game in enumerate(GAMES):
        g = scores[game]
        totals["me"]      += g["me"]
        totals["contact"] += g["contact"]
        totals["tie"]     += g["tie"]
        played = g["me"] + g["contact"] + g["tie"]

        with cols[i]:
            st.markdown(f"### {GAME_ICONS[game]} {game}")
            if played == 0:
                st.caption("No shared games yet.")
            else:
                if g["me"] > g["contact"]:
                    leader = "Iván leads 🏆"
                elif g["contact"] > g["me"]:
                    leader = f"{contact_first} leads 🏆"
                else:
                    leader = "Tied 🤝"
                st.caption(f"{played} games · {leader}")

                c1, c2 = st.columns(2)
                c1.metric("Iván", g["me"])
                c2.metric(contact_first, g["contact"])
                if g["tie"] > 0:
                    st.caption(f"🤝 {g['tie']} tie{'s' if g['tie'] > 1 else ''}")

    # ── Aggregate score ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 🏆 Total Score")

    tc = st.columns(3)
    tc[0].metric("Iván", totals["me"])
    tc[1].metric(contact_first, totals["contact"])
    tc[2].metric("Ties", totals["tie"])

    overall_played = totals["me"] + totals["contact"] + totals["tie"]
    if overall_played == 0:
        st.info("No head-to-head games found for this contact.")
    elif totals["me"] > totals["contact"]:
        st.success("Iván is winning overall! 🎉")
    elif totals["contact"] > totals["me"]:
        st.warning(f"{contact_first} is winning overall!")
    else:
        st.info("It's a draw overall! 🤝")

    # ── Detailed match history ─────────────────────────────────────────────────
    with st.expander("📋 Full match history"):
        any_duels = False
        for game in GAMES:
            duels = scores[game]["duels"]
            if not duels:
                continue
            any_duels = True
            st.markdown(f"**{GAME_ICONS[game]} {game}**")
            duel_df = pd.DataFrame(duels)
            duel_df["my_time"]      = duel_df["my_time"].apply(fmt_time)
            duel_df["contact_time"] = duel_df["contact_time"].apply(fmt_time)
            duel_df["winner"]       = duel_df["winner"].map(
                {"me": "Iván", "contact": contact_first, "tie": "Tie 🤝"}
            )
            duel_df.columns = ["Puzzle #", "Iván", contact_first, "Winner"]
            st.dataframe(duel_df, use_container_width=True, hide_index=True)
        if not any_duels:
            st.caption("No shared puzzles found for this contact.")


if __name__ == "__main__":
    main()
