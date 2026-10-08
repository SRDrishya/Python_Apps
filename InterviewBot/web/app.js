const startButton = document.querySelector("#start-button");
const pauseButton = document.querySelector("#pause-button");
const stopButton = document.querySelector("#stop-button");
const statusText = document.querySelector("#status");
const statusIndicator = document.querySelector("#status-indicator");
const transcriptPanel = document.querySelector("#transcript-panel");
const transcript = document.querySelector("#transcript");
const remoteAudio = document.querySelector("#remote-audio");

let peerConnection;
let microphone;
let eventChannel;
let currentEntry;
let isPaused = false;
let pendingAudioAction;

function setStatus(message, state = "") {
  statusText.textContent = message;
  statusIndicator.classList.toggle("is-active", state === "active");
  statusIndicator.classList.toggle("is-error", state === "error");
}

function resumeAudioPlayback() {
  remoteAudio.play().catch(() => {
    setStatus("Interview resumed. Allow audio playback to hear the interviewer.");
  });
}

function addTranscriptFragment(speaker, fragment) {
  if (!fragment) return;
  transcriptPanel.hidden = false;

  if (!currentEntry || currentEntry.dataset.speaker !== speaker) {
    currentEntry = document.createElement("p");
    currentEntry.className = `transcript-entry ${speaker}`;
    currentEntry.dataset.speaker = speaker;
    const label = document.createElement("strong");
    label.textContent = speaker === "assistant" ? "Interviewer" : "You";
    currentEntry.append(label, document.createTextNode(""));
    transcript.append(currentEntry);
  }

  currentEntry.lastChild.textContent += fragment;
  transcript.scrollTop = transcript.scrollHeight;
}

function handleServerEvent(event) {
  if (event.type === "session.started") {
    setStatus("Connected — the interviewer is starting", "active");
    pauseButton.disabled = false;
    stopButton.disabled = false;
    eventChannel.send(JSON.stringify({
      type: "session.instructions.append",
      delegation_id: null,
      content: "The candidate has joined. Delegate to the configured Responses backend now for the opening interview turn. Speak its result: briefly welcome them, ask what role they are preparing for, and ask for a short summary of their experience. Then stop and listen. For every later candidate turn, delegate before replying and speak the backend's response without inventing or replacing its interview question.",
    }));
  } else if (event.type === "session.input_transcript.delta") {
    addTranscriptFragment("candidate", event.delta);
  } else if (event.type === "session.output_transcript.delta") {
    addTranscriptFragment("assistant", event.delta);
  } else if (event.type === "session.instructions.appended") {
    setStatus("Interview in progress", "active");
  } else if (
    event.type === "session.input_audio.muted" ||
    event.type === "session.input_audio.unmuted"
  ) {
    const action = event.type === "session.input_audio.muted" ? "pause" : "resume";
    if (event.client_event_id !== pendingAudioAction?.eventId) return;
    window.clearTimeout(pendingAudioAction.timeoutId);
    pendingAudioAction = null;

    if (action === "pause") {
      isPaused = true;
      pauseButton.textContent = "Resume interview";
      pauseButton.disabled = false;
      setStatus("Interview paused");
      return;
    }

    isPaused = false;
    microphone?.getAudioTracks().forEach((track) => {
      track.enabled = true;
    });
    pauseButton.textContent = "Pause interview";
    pauseButton.disabled = false;
    resumeAudioPlayback();
    setStatus("Interview in progress", "active");
  } else if (event.type === "session.closed") {
    stopInterview("Interview ended", false);
  } else if (event.type === "error") {
    if (event.error?.client_event_id === pendingAudioAction?.eventId) {
      window.clearTimeout(pendingAudioAction.timeoutId);
      const failedAction = pendingAudioAction.action;
      pendingAudioAction = null;
      pauseButton.disabled = false;
      if (failedAction === "pause") {
        microphone?.getAudioTracks().forEach((track) => {
          track.enabled = true;
        });
        resumeAudioPlayback();
        isPaused = false;
        pauseButton.textContent = "Pause interview";
      }
    }
    const message = event.error?.message || "The live session returned an error.";
    setStatus(message, "error");
  }
}

function toggleInterviewPause() {
  if (eventChannel?.readyState !== "open" || pendingAudioAction) return;

  const action = isPaused ? "resume" : "pause";
  const eventId = `interview_${action}_${Date.now()}`;
  pauseButton.disabled = true;

  if (action === "pause") {
    microphone?.getAudioTracks().forEach((track) => {
      track.enabled = false;
    });
    remoteAudio.pause();
    isPaused = true;
    pauseButton.textContent = "Resume interview";
  }

  setStatus(action === "pause" ? "Pausing interview…" : "Resuming interview…");
  const timeoutId = window.setTimeout(() => {
    if (pendingAudioAction?.eventId !== eventId) return;
    const failedAction = pendingAudioAction.action;
    pendingAudioAction = null;
    pauseButton.disabled = false;
    if (failedAction === "pause") {
      microphone?.getAudioTracks().forEach((track) => {
        track.enabled = true;
      });
      resumeAudioPlayback();
      isPaused = false;
      pauseButton.textContent = "Pause interview";
    }
    setStatus("Could not confirm the pause state. Please try again.", "error");
  }, 10000);
  pendingAudioAction = { action, eventId, timeoutId };
  eventChannel.send(JSON.stringify({
    type: action === "pause" ? "session.input_audio.mute" : "session.input_audio.unmute",
    event_id: eventId,
  }));
}

function waitForIceGathering(connection) {
  if (connection.iceGatheringState === "complete") return Promise.resolve();

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out while preparing the browser connection."));
    }, 10000);

    function onStateChange() {
      if (connection.iceGatheringState === "complete") {
        cleanup();
        resolve();
      }
    }

    function cleanup() {
      window.clearTimeout(timeout);
      connection.removeEventListener("icegatheringstatechange", onStateChange);
    }

    connection.addEventListener("icegatheringstatechange", onStateChange);
  });
}

function stopInterview(message = "Ready when you are", closeSession = true) {
  if (closeSession && eventChannel?.readyState === "open") {
    eventChannel.send(JSON.stringify({ type: "session.close" }));
  }
  eventChannel?.close();
  if (pendingAudioAction) {
    window.clearTimeout(pendingAudioAction.timeoutId);
    pendingAudioAction = null;
  }
  peerConnection?.close();
  microphone?.getTracks().forEach((track) => track.stop());
  remoteAudio.srcObject = null;
  eventChannel = null;
  peerConnection = null;
  microphone = null;
  currentEntry = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  pauseButton.disabled = true;
  pauseButton.textContent = "Pause interview";
  isPaused = false;
  setStatus(message);
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  pauseButton.disabled = true;
  transcript.replaceChildren();
  transcriptPanel.hidden = true;
  setStatus("Connecting to the interview room…");

  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) {
      throw new Error("This browser does not support WebRTC microphone access.");
    }

    peerConnection = new RTCPeerConnection();
    peerConnection.addEventListener("track", (event) => {
      remoteAudio.srcObject = event.streams[0] || new MediaStream([event.track]);
      remoteAudio.play().catch(() => {
        setStatus("Allow audio playback in your browser to hear the interviewer.");
      });
    });
    peerConnection.addEventListener("connectionstatechange", () => {
      if (peerConnection?.connectionState === "failed") {
        setStatus("The browser connection failed. Please try again.", "error");
      }
    });

    microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
    microphone.getAudioTracks().forEach((track) => {
      peerConnection.addTrack(track, microphone);
    });

    eventChannel = peerConnection.createDataChannel("oai-events");
    eventChannel.addEventListener("message", ({ data }) => {
      try {
        handleServerEvent(JSON.parse(data));
      } catch (error) {
        setStatus(`Could not read a live session event: ${error.message}`, "error");
      }
    });
    eventChannel.addEventListener("open", () => {
      setStatus("Connected — waiting for the interviewer", "active");
    });
    eventChannel.addEventListener("error", () => {
      setStatus("The interview event channel encountered an error.", "error");
    });

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    await waitForIceGathering(peerConnection);

    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: peerConnection.localDescription.sdp }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Could not create the live interview.");
    }

    await peerConnection.setRemoteDescription({
      type: "answer",
      sdp: result.transport.sdp,
    });
  } catch (error) {
    const message = error instanceof TypeError && error.message === "Failed to fetch"
      ? "Can't reach InterviewBot. Start the server with `python main.py` from the InterviewBot folder, then reload this page."
      : error.message || "Could not start the interview";
    stopInterview(message);
    statusIndicator.classList.add("is-error");
  }
});

pauseButton.addEventListener("click", toggleInterviewPause);
stopButton.addEventListener("click", () => stopInterview("Interview ended"));
