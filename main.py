from dataclasses import dataclass
from datetime import datetime
import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import fire
@dataclass
class GeneralStats:
    time_taken: float

@dataclass
class KeyAnalysis:
    total_changes: int
    erroneous_changes: int
    non_contiguous_errors_per_100_chars: float
    message_length: int

def analyze_general_stats(df: pd.DataFrame) -> GeneralStats:
    # find session_start message
    session_start = df[df['message'] == 'session_start']
    start_time_str = session_start['timestamp'].iloc[0]
    # Replace 'Z' with '+00:00' for ISO format compatibility
    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

    # find session_end message
    session_end = df[df['message'] == 'session_end']
    end_time_str = session_end['timestamp'].iloc[0]
    # Replace 'Z' with '+00:00' for ISO format compatibility
    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))

    # Calculate time difference as total seconds
    time_difference = (end_time - start_time).total_seconds()
    
    return GeneralStats(time_taken=time_difference)


def task_code_to_task_name(session_code: str) -> str:
    if session_code in ['1', '1p']:
        return 'Task 1'
    elif session_code in ['2', '2p', '2u']:
        return 'Task 2'
    elif session_code in ['31', '31p', '31u']:
        return 'Task 3.1'
    elif session_code in ['32', '32p', '32u']:
        return 'Task 3.2'
    
    raise ValueError(f"Invalid session code: {session_code}")

def analyze_key_changes(user_id, session_code, df: pd.DataFrame) -> KeyAnalysis:
    total_changes = 0
    erroneous_changes = 0
    non_contiguous_errors = 0
    final_length = 0
    
    keystrokes = []
    lastDeleted = None
    lastOutcomeChanged = None
    contiguousBackspace = False

    group_sorted = df.sort_values('timestamp')
    for idx, row in group_sorted.iterrows():
        if row['message'] == 'changed_outcome':
            lastOutcomeChanged = json.loads(row['data'])
            total_changes += 1

        elif row['message'] == 'key_typed':
            data = json.loads(row['data'])
            keystrokes.append({
                'timestamp': row['timestamp'],
                'keystroke': data['key'],
                'outcomeChanged': lastOutcomeChanged
            })
            final_length = len(data['currentInput']) # Update final length
            if lastDeleted is not None:
                if lastDeleted['outcomeChanged'] is not None and lastDeleted['outcomeChanged'].get('changedFrom') == data['key']:
                    erroneous_changes += 1
                lastDeleted = None
            contiguousBackspace = False
            lastOutcomeChanged = None

        elif row['message'] == 'backspace_pressed':
            if keystrokes:
                lastDeleted = keystrokes.pop()
                if not contiguousBackspace:
                    non_contiguous_errors += 1
                contiguousBackspace = True

    # Calculate non-contiguous errors per 100 characters
    errors_per_100 = (non_contiguous_errors / final_length) * 100 if final_length > 0 else 0
    
    return KeyAnalysis(
        total_changes=total_changes, 
        erroneous_changes=erroneous_changes, 
        non_contiguous_errors_per_100_chars=errors_per_100,
        message_length=final_length
    )

def analyze_sessions(df):
    """
    Compute per-session metrics (total changes, erroneous changes, and non-contiguous errors)
    and return a list of dictionaries with per-session statistics.
    """    
    session_stats = {}

    # Group by user and session to calculate per-session metrics
    for (user_id, session_code), group in df.groupby(['userId', 'sessionCode']):
        key_changes = analyze_key_changes(user_id, session_code, group)
        general_stats = analyze_general_stats(group)
        session_stats[(user_id, session_code)] = {
            "key_changes": key_changes,
            "general_stats": general_stats
        }

    return session_stats


def main(csvPath):
    df = pd.read_csv(csvPath)
    session_stats = analyze_sessions(df)
    phone_df = pd.read_csv("data-clean/phone-time-taken.csv")

    # Accumulate “long” records
    records = []
    def add(user_id, session_code, task_name, modality, stats):
        rec = {
            "user_id": user_id,
            "task_name": task_name,
            "modality": modality,
            "time_taken": round(stats["general_stats"].time_taken, 3),
            "message_length": stats["key_changes"].message_length if modality.startswith("smartwatch") else None,
            "total_changes": stats["key_changes"].total_changes if modality=="smartwatch_assisted" else None,
            "erroneous_changes": stats["key_changes"].erroneous_changes if modality=="smartwatch_assisted" else None,
            "non_contiguous_errors_per_100_chars": round(stats["key_changes"].non_contiguous_errors_per_100_chars,3)
                if modality.startswith("smartwatch") else None
        }
        records.append(rec)

    # Populate assisted/unassisted
    for (uid, sc), stats in session_stats.items():
        base = sc.rstrip("up")  # strip trailing 'p' or 'u'
        task = task_code_to_task_name(base)
        if sc.endswith("u"):
            add(uid, sc, task, "smartwatch_unassisted", stats)
        elif sc.endswith("p"):
            # phone will be handled separately
            continue
        else:
            add(uid, sc, task, "smartwatch_assisted", stats)

    # Populate phone
    for _, row in phone_df.iterrows():
        uid = row["u"]
        task = task_code_to_task_name(str(int(row["task"])))
        records.append({
            "user_id": uid,
            "task_name": task,
            "modality": "phone",
            "time_taken": round(row["time_taken"], 3),
            "message_length": None,
            "total_changes": None,
            "erroneous_changes": None,
            "non_contiguous_errors_per_100_chars": None
        })

    long_df = pd.DataFrame(records)

    assist_cols = [
        "user_id", "task_name",
        "time_taken", "message_length",
        "total_changes", "erroneous_changes",
        "non_contiguous_errors_per_100_chars",
    ]
    df_assist = (
        long_df[long_df.modality == "smartwatch_assisted"][assist_cols]
        .set_index(["user_id","task_name"])
        .rename(columns={
            "time_taken": "smartwatch_assisted_time_taken",
            "message_length": "smartwatch_assisted_message_length",
            "total_changes": "smartwatch_assisted_total_changes",
            "erroneous_changes": "smartwatch_assisted_erroneous_changes",
            "non_contiguous_errors_per_100_chars":
                "smartwatch_assisted_non_contiguous_errors_per_100_chars"
        })
    )

    # 2) Unassisted — only keep the columns you actually want
    unassist_cols = [
        "user_id", "task_name",
        "time_taken", "message_length",
        "non_contiguous_errors_per_100_chars",
    ]
    df_unassist = (
        long_df[long_df.modality == "smartwatch_unassisted"][unassist_cols]
        .set_index(["user_id","task_name"])
        .rename(columns={
            "time_taken": "smartwatch_unassisted_time_taken",
            "message_length": "smartwatch_unassisted_message_length",
            "non_contiguous_errors_per_100_chars":
                "smartwatch_unassisted_non_contiguous_errors_per_100_chars"
        })
    )

    # 3) Phone — again, drop everything except what you need
    phone_cols = ["user_id","task_name","time_taken"]
    df_phone = (
        long_df[long_df.modality == "phone"][phone_cols]
        .set_index(["user_id","task_name"])
        .rename(columns={"time_taken": "phone_time_taken"})
    )

    # 4) Merge
    merged = (
        df_assist
        .join(df_unassist, how="outer")
        .join(df_phone,   how="outer")
        .reset_index()
        .sort_values(["user_id","task_name"])
    )

    merged.to_csv("session-metrics.csv", index=False)

if __name__ == "__main__":
    fire.Fire(main)
