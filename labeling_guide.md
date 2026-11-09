# DriveAware Labeling Guide: Gemini label sheet (Model B)

Owner: Bindu. Second checker: Mahitha (50-row check).
Use this guide for every row of the Gemini label sheet. Each row is one triggered event: a 512 px face keyframe plus a signal summary. You write the answer we want the fine-tuned Gemini to give.

Thresholds below come from the approved proposal (Section 3). Signal field names match the signal line in driveaware-app `backend/app/logic.py` (`build_prompt`). If Vanessa's tuned values in thresholds.csv change them, update this guide before labeling more rows.

## 1. Columns and allowed words

Use these words exactly. Lowercase, no spaces, no other values. Swathi's JSON schema is built from this list.

| Column | Allowed values |
|---|---|
| event | drowsy, distracted_phone, distracted_gaze, drinking, smoking, early_fatigue, false_alarm |
| confirmed | yes, no |
| severity | low, medium, high |
| message | one calm sentence, 20 words or fewer |
| action | none, refocus, take_break_soon, pull_over_now |
| labeler, checked_by | your first name |

**confirmed:** `no` only when event is `false_alarm`. Every other event is `yes`.

## 2. The authority rule (most important)

When the trigger is a **measured signal**, the answer is **never `false_alarm`**:

- eye closure of 1.0 s or more (`eye_closure_long`, or `longest_closure_s` ≥ 1.0)
- PERCLOS above threshold (`perclos_high`, `perclos` > 0.15)
- nodding (`nod`, or `nods_5min` ≥ 1)

Even if the keyframe shows open eyes, trust the signal. A single frame can miss a closure.

Use `false_alarm` **only for ambiguous visual events**:

- Model A activities that the keyframe shows are wrong (hand near ear but scratching, cup confused with phone, pen that is clearly not held like a cigarette)
- gaze-away that looks like a normal mirror or shoulder check

## 3. Choosing the event

| Trigger | Label as | Use false_alarm when |
|---|---|---|
| eye_closure_long, perclos_high, nod | drowsy | never (authority rule) |
| yawn, face_touch (eye rubs) | early_fatigue, or drowsy if section 4 says so | the keyframe clearly shows talking, eating or a one-off scratch with no other fatigue signs |
| phone_use | distracted_phone | no phone visible, or hand is clearly doing something else |
| gaze_away | distracted_gaze | it looks like a mirror or shoulder check |
| drinking | drinking | no cup or bottle at or near the mouth |
| smoking | smoking | no cigarette, vape or smoking prop in hand or mouth |

If two things happen at once, label the **more dangerous** one: drowsy > distracted_phone > distracted_gaze > early_fatigue > drinking = smoking.

## 4. Severity and action rules

| Event | Condition | severity | action |
|---|---|---|---|
| drowsy | any eye closure ≥ 1.0 s, or a nod while PERCLOS > 0.15 | high | pull_over_now |
| drowsy | PERCLOS > 0.15 (with or without yawns), or a nod alone | medium | take_break_soon |
| drowsy | 3+ yawns in 5 min **and** PERCLOS > 0.15 | medium | take_break_soon |
| early_fatigue | 1 to 2 yawns in 5 min, PERCLOS ≤ 0.15, no closure ≥ 1.0 s | low | take_break_soon |
| early_fatigue | 3+ yawns in 5 min, or 3+ eye rubs in 5 min, PERCLOS ≤ 0.15 | medium | take_break_soon |
| distracted_phone | phone held to ear | medium | refocus |
| distracted_phone | texting or looking at phone in hand | high | refocus |
| distracted_gaze | eyes or head off road 2 to 4 s | low | refocus |
| distracted_gaze | eyes or head off road more than 4 s | medium | refocus |
| drinking | cup or bottle at the mouth | low | refocus |
| smoking | cigarette, vape or prop in hand or mouth | low | refocus |
| false_alarm | any | low | none |

**Sunglasses (`sunglasses=yes`):** ignore `perclos` and `longest_closure_s`, because the eyes can't be measured. Judge fatigue only from yawns, nods and eye rubs. Never label `high` from eye signals alone. The message should not claim the driver's eyes closed.

## 5. Writing the message

- One sentence, 20 words or fewer, ending with a period.
- Calm, second person ("you"), no blame, no exclamation marks, no emojis.
- Say what was noticed or what to do, and match the action: `pull_over_now` → tell them to pull over safely; `take_break_soon` → suggest a break; `refocus` → eyes or hands back on the road.
- For `false_alarm`, use exactly: `No alert needed.`
- Don't guess at causes (no "you seem drunk", "you look sick").

## 6. Worked examples

These signal lines are **made-up examples** to show how the rules apply. They are not real data.

| # | Signal summary (and keyframe) | event | confirmed | severity | action | message |
|---|---|---|---|---|---|---|
| 1 | trigger=eye_closure_long longest_closure_s=1.6 perclos=0.12 sunglasses=no (keyframe: eyes open) | drowsy | yes | high | pull_over_now | Your eyes closed for over a second. Please pull over safely and rest. |
| 2 | trigger=perclos_high perclos=0.21 yawns_5min=3 longest_closure_s=0.8 sunglasses=no | drowsy | yes | medium | take_break_soon | You're showing signs of fatigue. Plan a break in the next few minutes. |
| 3 | trigger=nod nods_5min=1 perclos=0.09 sunglasses=no | drowsy | yes | medium | take_break_soon | Your head dropped forward. Consider taking a break soon. |
| 4 | trigger=yawn yawns_5min=1 perclos=0.06 sunglasses=no | early_fatigue | yes | low | take_break_soon | You're yawning. Stay alert and plan a break if you feel tired. |
| 5 | trigger=yawn yawns_5min=3 perclos=0.10 sunglasses=yes | early_fatigue | yes | medium | take_break_soon | You've yawned several times. A short break soon would help. |
| 6 | trigger=phone_use model_a=phone_use:0.84 hand_occupied=yes (keyframe: texting, phone in lap) | distracted_phone | yes | high | refocus | Put the phone down and keep your eyes on the road. |
| 7 | trigger=phone_use model_a=phone_use:0.71 (keyframe: scratching ear, no phone) | false_alarm | no | low | none | No alert needed. |
| 8 | trigger=gaze_away gaze_away_s=5.0 (keyframe: looking at passenger seat) | distracted_gaze | yes | medium | refocus | Eyes back on the road, please. |
| 9 | trigger=gaze_away gaze_away_s=2.3 (keyframe: head turned to side mirror) | false_alarm | no | low | none | No alert needed. |
| 10 | trigger=drinking model_a=drinking:0.78 (keyframe: bottle at mouth) | drinking | yes | low | refocus | Keep at least one hand on the wheel while you drink. |
