import streamlit as st
import pandas as pd
import re

# ── Configuration ──────────────────────────────────────────────────────────────
MY_NAME_DEFAULT = "Iván Nicolás Gutiérrez Arias"

GAMES = ["Tango", "Queens", "Zip", "Mini Sudoku"]

GAME_ICONS = {
    "Tango":       "🟡",
    "Queens":      "👑",
    "Zip":         "⚡",
    "Mini Sudoku": "🔢",
}

# Matches all four game formats:
#   "Tango n.º 240 | 0:46"      "Queens n.º 400 | 1:32"
#   "Mini Sudoku n.º 193 | 1:41" "Zip #78 | 0:13"
# Mini Sudoku must appear first to avoid partial matches.
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
    Combine CSV and conversation results.
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

    if "manual_results" not in st.session_state:
        st.session_state.manual_results = _EMPTY_RESULTS.copy()

    st.title("🎮 LinkedIn Games Tracker")
    st.caption("Compare your LinkedIn mini-game scores against your contacts.")

    my_name  = st.text_input(
        "Your LinkedIn name (exactly as it appears in messages)",
        value=MY_NAME_DEFAULT,
        key="my_name",
    ).strip()
    my_first = my_name.split()[0] if my_name else "Me"

    if not my_name:
        st.warning("Enter your LinkedIn name above to get started.")
        return

    # ── Data input tabs ────────────────────────────────────────────────────────
    tab_csv, tab_convo = st.tabs(["📁 CSV Export", "💬 Conversation"])

    csv_results = _EMPTY_RESULTS.copy()

    # Tab 1: Full LinkedIn CSV export ──────────────────────────────────────────
    with tab_csv:
        st.markdown(
            "Upload your full LinkedIn message history. "
            "**How to export:** LinkedIn → Settings → Data Privacy → "
            "Get a copy of your data → Messages"
        )
        uploaded = st.file_uploader("Choose `messages.csv`", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                missing_cols = {"FROM", "CONTENT", "DATE"} - {c.upper() for c in df.columns}
                if missing_cols:
                    st.error(
                        f"Missing columns: **{missing_cols}**\n\n"
                        f"Columns in your file: `{list(df.columns)}`"
                    )
                else:
                    csv_results = parse_messages(df)
                    if csv_results.empty:
                        st.warning("No game results found in the CSV.")
                    else:
                        st.success(f"Loaded {len(csv_results)} game results from CSV.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    # Tab 2: Individual conversation (paste or .txt upload) ────────────────────
    with tab_convo:
        st.markdown(
            "Open any LinkedIn conversation, select all (Ctrl+A / Cmd+A), copy, "
            "then either paste below or save it as a `.txt` file and upload it. "
            "Speaker attribution is detected automatically from the message headers."
        )

        contact_input = st.text_input(
            "Contact's full name (exactly as it appears in LinkedIn)",
            placeholder="e.g. Jorge Herrera Toro",
            key="convo_contact",
        )

        txt_file = st.file_uploader("Upload `.txt` file", type=["txt"], key="txt_upload")
        convo_paste = st.text_area(
            "Or paste the conversation directly",
            height=180,
            key="convo_paste",
            placeholder="Paste the full LinkedIn conversation here…",
        )

        # Prefer the uploaded file; fall back to the text area
        if txt_file is not None:
            convo_text = txt_file.getvalue().decode("utf-8", errors="replace")
            st.caption(f"Using file: **{txt_file.name}**")
        else:
            convo_text = convo_paste

        btn_add, btn_clear, _ = st.columns([1, 1, 4])

        if btn_add.button("Process & Add", type="primary", key="btn_add"):
            if not contact_input.strip():
                st.error("Enter the contact's full name first.")
            elif not convo_text.strip():
                st.warning("No conversation text provided.")
            else:
                records, names_detected = parse_conversation(
                    convo_text, my_name, contact_input.strip()
                )
                if not names_detected:
                    st.error(
                        f"Could not find **{my_name}** or **{contact_input.strip()}** "
                        "in the text. Make sure the names match exactly as they appear "
                        "in LinkedIn."
                    )
                elif not records:
                    st.warning("Names were found but no game results were detected.")
                else:
                    my_count = sum(1 for r in records if r["sender"] == my_name)
                    co_count = len(records) - my_count
                    first    = contact_input.strip().split()[0]
                    st.session_state.manual_results = pd.concat(
                        [st.session_state.manual_results, pd.DataFrame(records)],
                        ignore_index=True,
                    )
                    st.success(
                        f"Added {len(records)} result{'s' if len(records) > 1 else ''}: "
                        f"{my_count} for {my_first}, {co_count} for {first}."
                    )
                    st.rerun()

        if btn_clear.button("Clear all", key="btn_clear"):
            st.session_state.manual_results = _EMPTY_RESULTS.copy()
            st.rerun()

        # Summary of conversations already loaded
        if not st.session_state.manual_results.empty:
            st.divider()
            st.caption("**Conversations loaded:**")
            summary = (
                st.session_state.manual_results[
                    st.session_state.manual_results["sender"] != my_name
                ]
                .groupby("sender")
                .size()
                .reset_index(name="n")
            )
            for _, row in summary.iterrows():
                st.caption(f"• {row['sender']}: {row['n']} results")

    # ── Merge all sources ──────────────────────────────────────────────────────
    results = merge_results(csv_results, st.session_state.manual_results)

    if results.empty:
        st.info("Add data above to get started — upload a CSV or load a conversation.")
        return

    if my_name not in results["sender"].unique():
        st.error(
            f"Your name **{my_name}** was not found in the data.\n\n"
            "Make sure you're uploading your own LinkedIn CSV export, or that "
            "your name appears in the pasted conversation."
        )
        return

    contacts = sorted(s for s in results["sender"].unique() if s != my_name)

    if not contacts:
        st.warning("No contacts with game results found.")
        return

    # ── Sidebar: overview ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Overview")
        st.metric("CSV results",    len(csv_results))
        st.metric("Manual results", len(st.session_state.manual_results))
        st.metric("Total",          len(results))
        st.metric("Contacts",       len(contacts))
        st.markdown("**Results by game**")
        st.bar_chart(results["game"].value_counts())

    # ── Contact selector ───────────────────────────────────────────────────────
    st.divider()
    contact = st.selectbox("🔍 Compare against:", contacts)
    if not contact:
        return

    scores        = compute_scores(results, my_name, contact)
    contact_first = contact.split()[0]

    # ── Per-game scorecards ────────────────────────────────────────────────────
    st.markdown(f"## {my_first} vs {contact}")
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
                    leader = f"{my_first} leads 🏆"
                elif g["contact"] > g["me"]:
                    leader = f"{contact_first} leads 🏆"
                else:
                    leader = "Tied 🤝"
                st.caption(f"{played} games · {leader}")

                c1, c2 = st.columns(2)
                c1.metric(my_first, g["me"])
                c2.metric(contact_first, g["contact"])
                if g["tie"] > 0:
                    st.caption(f"🤝 {g['tie']} tie{'s' if g['tie'] > 1 else ''}")

    # ── Aggregate score ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 🏆 Total Score")

    tc = st.columns(3)
    tc[0].metric(my_first, totals["me"])
    tc[1].metric(contact_first, totals["contact"])
    tc[2].metric("Ties", totals["tie"])

    overall_played = totals["me"] + totals["contact"] + totals["tie"]
    if overall_played == 0:
        st.info("No head-to-head games found for this contact.")
    elif totals["me"] > totals["contact"]:
        st.success(f"{my_first} is winning overall! 🎉")
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
                {"me": my_first, "contact": contact_first, "tie": "Tie 🤝"}
            )
            duel_df.columns = ["Puzzle #", my_first, contact_first, "Winner"]
            st.dataframe(duel_df, use_container_width=True, hide_index=True)
        if not any_duels:
            st.caption("No shared puzzles found for this contact.")


if __name__ == "__main__":
    main()
