import streamlit as st
import pandas as pd
import re

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_SIZE_LIMIT_MB = 50   # LinkedIn personal exports are typically < 5 MB
TXT_SIZE_LIMIT_MB = 2

GAMES = ["Tango", "Queens", "Zip", "Mini Sudoku"]

GAME_ICONS = {
    "Tango":       "🟡",
    "Queens":      "👑",
    "Zip":         "⚡",
    "Mini Sudoku": "🔢",
}

# Matches all four game formats:
#   "Tango n.º 240 | 0:46"       "Queens n.º 400 | 1:32"
#   "Mini Sudoku n.º 193 | 1:41"  "Zip #78 | 0:13"
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
_MD_SPECIAL = re.compile(r"[*_`\[\]()#<>!|\\]")

def safe_md(text: str) -> str:
    """Strip Markdown special characters from user-derived strings before rendering."""
    return _MD_SPECIAL.sub("", text)


def to_seconds(minutes: str, seconds: str) -> int:
    return int(minutes) * 60 + int(seconds)


def fmt_time(total_seconds: int) -> str:
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def detect_speakers(text: str) -> list[str]:
    """
    Extract unique speaker names from a LinkedIn conversation.

    LinkedIn message headers have the format  "Full Name   HH:MM"  — the name
    is followed by 2+ spaces then the time at the end of the line.  This is
    distinct from the transition line "Name ha enviado … a las HH:MM", which
    only has a single space before the time, so it is correctly ignored.
    """
    pattern = re.compile(r"^(.+?)\s{2,}\d{1,2}:\d{2}\s*$", re.MULTILINE)
    seen: set[str] = set()
    names: list[str] = []
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def detect_my_name_from_csv(df: pd.DataFrame) -> str | None:
    """
    Detect the app user's name from a LinkedIn CSV export.

    The user is the sender who appears in the most distinct Conversation IDs —
    they participate in every conversation, while each contact only appears in one.
    """
    col_map  = {c.upper(): c for c in df.columns}
    from_col  = col_map.get("FROM")
    convo_col = col_map.get("CONVERSATION ID")
    if not from_col or not convo_col:
        return None
    counts = df.groupby(from_col)[convo_col].nunique()
    return str(counts.idxmax()) if not counts.empty else None


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
    from_col    = col.get("FROM")
    content_col = col.get("CONTENT")
    date_col    = col.get("DATE")

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

    # ── Session state ──────────────────────────────────────────────────────────
    if "manual_results" not in st.session_state:
        st.session_state.manual_results = _EMPTY_RESULTS.copy()
    if "my_name" not in st.session_state:
        st.session_state.my_name = None

    st.title("🎮 LinkedIn Games Tracker")
    st.caption("Compare your LinkedIn mini-game scores against your contacts.")

    with st.expander("🔒 Privacy & data handling", expanded=False):
        st.markdown(
            "- **Nothing is stored.** Uploaded files are processed in memory and "
            "discarded when you close the tab.\n"
            "- **No third parties receive your data** when running locally. "
            "If hosted on Streamlit Community Cloud, files transit their servers "
            "over HTTPS but are not persisted.\n"
            "- The app only reads game-result lines from your messages — "
            "no other message content is used or displayed."
        )

    # Identity banner — shown once identity is known, with a Change button
    if st.session_state.my_name:
        id_col, change_col = st.columns([9, 1])
        id_col.caption(f"Showing results as: **{safe_md(st.session_state.my_name)}**")
        if change_col.button("Change", key="reset_identity"):
            st.session_state.my_name = None
            st.session_state.manual_results = _EMPTY_RESULTS.copy()
            st.rerun()

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
            if uploaded.size > CSV_SIZE_LIMIT_MB * 1024 * 1024:
                st.error(
                    f"File exceeds the {CSV_SIZE_LIMIT_MB} MB limit. "
                    "LinkedIn personal exports are typically under 5 MB."
                )
            else:
                try:
                    df = pd.read_csv(uploaded)
                    missing = {"FROM", "CONTENT", "DATE"} - {c.upper() for c in df.columns}
                    if missing:
                        st.error(
                            f"Missing columns: **{missing}**\n\n"
                            f"Columns found: `{list(df.columns)}`"
                        )
                    else:
                        # Auto-detect identity from conversation participation
                        if st.session_state.my_name is None:
                            detected = detect_my_name_from_csv(df)
                            if detected:
                                st.session_state.my_name = detected

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
            "then upload as a `.txt` file or paste below. "
            "Speakers are detected automatically from the message headers."
        )

        txt_file = st.file_uploader("Upload `.txt` file", type=["txt"], key="txt_upload")
        convo_paste = st.text_area(
            "Or paste the conversation directly",
            height=180,
            key="convo_paste",
            placeholder="Paste the full LinkedIn conversation here…",
        )

        if txt_file is not None and txt_file.size > TXT_SIZE_LIMIT_MB * 1024 * 1024:
            st.error(f"File exceeds the {TXT_SIZE_LIMIT_MB} MB limit.")
            txt_file = None

        convo_text = (
            txt_file.getvalue().decode("utf-8", errors="replace")
            if txt_file is not None
            else convo_paste
        )
        if txt_file is not None:
            st.caption(f"Using file: {safe_md(txt_file.name)}")

        if convo_text.strip():
            speakers = detect_speakers(convo_text)

            if not speakers:
                st.warning(
                    "Could not detect any speaker names. "
                    "Make sure the conversation includes message headers (e.g. "
                    "\"Full Name   8:41\")."
                )

            elif st.session_state.my_name is None:
                # ── Identity unknown: ask which speaker the user is ────────────
                st.markdown("**Detected speakers — which one are you?**")
                btn_cols = st.columns(min(len(speakers), 2))
                for i, sp in enumerate(speakers[:2]):
                    if btn_cols[i].button(sp, key=f"iam_{i}", use_container_width=True):
                        st.session_state.my_name = sp
                        st.rerun()

            else:
                # ── Identity known: contact is the other speaker ───────────────
                my_name = st.session_state.my_name
                contact = next((s for s in speakers if s != my_name), None)

                if contact is None:
                    st.warning(
                        f"Only **{safe_md(my_name.split()[0])}** was detected in this conversation — "
                        "no contact results to add."
                    )
                else:
                    st.caption(f"You: **{safe_md(my_name)}** · Contact: **{safe_md(contact)}**")

                    btn_add, btn_clear, _ = st.columns([1, 1, 4])

                    if btn_add.button("Process & Add", type="primary", key="btn_add"):
                        records, _ = parse_conversation(convo_text, my_name, contact)
                        if records:
                            my_count = sum(1 for r in records if r["sender"] == my_name)
                            co_count = len(records) - my_count
                            st.session_state.manual_results = pd.concat(
                                [st.session_state.manual_results, pd.DataFrame(records)],
                                ignore_index=True,
                            )
                            st.success(
                                f"Added {len(records)} result{'s' if len(records) > 1 else ''}: "
                                f"{my_count} for {safe_md(my_name.split()[0])}, "
                                f"{co_count} for {safe_md(contact.split()[0])}."
                            )
                            st.rerun()
                        else:
                            st.warning("No game results found in this conversation.")

                    if btn_clear.button("Clear all", key="btn_clear"):
                        st.session_state.manual_results = _EMPTY_RESULTS.copy()
                        st.rerun()

        # Summary of conversations already loaded
        if not st.session_state.manual_results.empty and st.session_state.my_name:
            st.divider()
            st.caption("**Conversations loaded:**")
            summary = (
                st.session_state.manual_results[
                    st.session_state.manual_results["sender"] != st.session_state.my_name
                ]
                .groupby("sender")
                .size()
                .reset_index(name="n")
            )
            for _, row in summary.iterrows():
                st.caption(f"• {row['sender']}: {row['n']} results")

    # ── Merge all sources ──────────────────────────────────────────────────────
    my_name = st.session_state.my_name

    if my_name is None:
        st.info("Upload a CSV or paste a conversation to get started.")
        return

    my_first = safe_md(my_name.split()[0])
    results  = merge_results(csv_results, st.session_state.manual_results)

    if results.empty:
        st.info("Add data above to get started — upload a CSV or load a conversation.")
        return

    if my_name not in results["sender"].unique():
        st.error(
            f"**{safe_md(my_name)}** was not found in the data.\n\n"
            "Make sure you're uploading your own LinkedIn CSV, or that your name "
            "appears in the pasted conversation."
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
    contact_first = safe_md(contact.split()[0])

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
