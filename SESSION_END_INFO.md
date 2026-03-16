# Session Termination Information

## Expected Behavior

When a voice session ends (either by user saying "goodbye" or clicking disconnect), you'll see these log messages:

### Normal Flow:
```
INFO: Session end initiated by end_session() call
INFO: Disconnecting session gracefully (cleanup warnings below are expected)
INFO: livekit::room - disconnected from room with reason: ClientInitiated
INFO: Session ended successfully
```

### Expected Cleanup Warnings:
These are NORMAL and expected during graceful shutdown:

```
ERROR: deepgram connection closed unexpectedly
WARNING: failed to recognize speech, retrying
WARN: error reading data channel
```

**Why these occur:**
- When the session ends, LiveKit/Deepgram connections are closed immediately
- Some async tasks are still trying to read/write when connections close
- These are caught and handled properly by the SDK
- No data is lost, session ends cleanly

## Summary

✅ **Session termination is working correctly**
✅ **Error messages are expected cleanup warnings**
✅ **Frontend properly receives disconnect event**
✅ **All resources are cleaned up**

The important log to look for is: `Session ended successfully`
