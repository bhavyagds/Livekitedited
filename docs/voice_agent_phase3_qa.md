# Elena Phase 3 QA Plan

This checklist is for validating the deterministic order/phone flow in English and Greek after Phase 3.

## 1) Local Automated Regression

Run:

```bash
python -m py_compile src/agents/elena.py src/agents/tools/order_lookup.py
python -m unittest tests.test_elena_regressions -v
```

Coverage includes:
- order id normalization
- strict Greek phone normalization
- not-found classification variants
- deterministic `not_found` and `unknown` summaries
- no wait-phrase duplication in lookup return path

## 2) End-to-End Call QA (EN + EL)

Run 6 calls (3 EN + 3 EL) and capture room logs/transcripts.

### Scenario A: Valid order id -> found
Expected:
- asks for order id
- calls `lookup_order`
- `SUPPORT_FLOW_STATE` moves to `checking_order_number` then `order_found`
- response includes concise status/date/total

### Scenario B: Invalid/missing order id
Expected:
- asks to repeat order id one digit at a time
- remains in `awaiting_order_number`
- does **not** switch to phone flow unless user explicitly says no order number

### Scenario C: No order number -> phone flow
Expected:
- switches to `awaiting_phone_number`
- captures phone
- asks confirmation (yes/no)
- only calls `lookup_order_by_phone` after yes

### Scenario D: Phone not found
Expected:
- deterministic not-found phone response
- `SUPPORT_FLOW_STATE` returns to `awaiting_phone_number`

### Scenario E: Phone confirmation = no
Expected:
- clears pending phone
- asks for phone again one digit at a time

### Scenario F: No duplicate speech
Expected:
- no duplicate lookup summaries
- no duplicated “one moment…” phrase in a single reply

## 3) Log Validation

Use:

```bash
python scripts/qa_check_flow_log.py --log <ROOM_LOG_PATH>
```

Check:
- `SUPPORT_FLOW_STATE` events exist
- both order and phone paths appear where expected
- no suspicious repeated wait-ack behavior

## 4) Production Observation (1-2 days)

Track:
- not-found rate (order / phone)
- phone confirmation drop-off rate (users saying no repeatedly)
- silent-call prompt rate
- support ticket escalation rate after lookup failures

Tune if needed:
- `phone_lookup_schedule_snooze_seconds`
- `invalid_number_recovery_silence_grace_seconds`
- `language_switch_min_turns`
- wait phrase toggles/strictness

## 5) Exit Criteria

- All automated regressions pass.
- EN and EL E2E scenarios pass with expected state transitions.
- No duplicate lookup speech incidents in logs.
- Stable production behavior for 1-2 days without new lookup regressions.
