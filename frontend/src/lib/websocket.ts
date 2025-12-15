// frontend/src/lib/websocket.ts

import type { WSMessage, WSClientMessage } from '@/types';

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

type MessageHandler = (message: WSMessage) => void;
type ConnectionHandler = () => void;
type ErrorHandler = (error: Event) => void;

export class InterviewWebSocket {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private messageHandlers: MessageHandler[] = [];
  private connectHandlers: ConnectionHandler[] = [];
  private disconnectHandlers: ConnectionHandler[] = [];
  private errorHandlers: ErrorHandler[] = [];

  constructor(sessionId: string) {
    this.sessionId = sessionId;
    console.log('[WS] Created for session:', sessionId);
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${WS_BASE_URL}/ws/interview/${this.sessionId}`;
      console.log('[WS] Connecting to:', url);

      try {
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log('[WS] Connection opened');
          this.connectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onmessage = (event) => {
          console.log('[WS] Raw message received:', event.data);
          try {
            const message: WSMessage = JSON.parse(event.data);
            console.log('[WS] Parsed message:', message);
            this.messageHandlers.forEach((handler) => handler(message));
          } catch (error) {
            console.error('[WS] Failed to parse message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('[WS] Connection closed:', event.code, event.reason);
          this.disconnectHandlers.forEach((handler) => handler());
        };

        this.ws.onerror = (error) => {
          console.error('[WS] Connection error:', error);
          this.errorHandlers.forEach((handler) => handler(error));
          reject(error);
        };
      } catch (error) {
        console.error('[WS] Failed to create WebSocket:', error);
        reject(error);
      }
    });
  }

  disconnect(): void {
    console.log('[WS] Disconnecting');
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message: WSClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const json = JSON.stringify(message);
      console.log('[WS] Sending:', json);
      this.ws.send(json);
    } else {
      console.error('[WS] Cannot send - not connected. ReadyState:', this.ws?.readyState);
    }
  }

  // Convenience methods
  startInterview(numQuestions: number = 5, voiceEnabled: boolean = true): void {
    console.log('[WS] Starting interview with', numQuestions, 'questions');
    this.send({ type: 'start', data: { num_questions: numQuestions, voice_enabled: voiceEnabled} });
  }

  sendAnswer(answer: string, voiceEnabled: boolean = true): void {
    this.send({ type: 'answer', content: answer, data: {voice_enabled: voiceEnabled}});
  }

  sendAudioAnswer(audioBase64: string, voiceEnabled: boolean = true): void {
    this.send({ type: 'answer_audio', content: audioBase64, data: {voice_enabled: voiceEnabled}});
  }

  skipQuestion(voiceEnabled: boolean = true): void {
    this.send({ type: 'skip', data: {voice_enabled: voiceEnabled} });
  }

  endInterview(): void {
    this.send({ type: 'end' });
  }

  ping(): void {
    this.send({ type: 'ping' });
  }

  // Event handlers
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.push(handler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter((h) => h !== handler);
    };
  }

  onConnect(handler: ConnectionHandler): () => void {
    this.connectHandlers.push(handler);
    return () => {
      this.connectHandlers = this.connectHandlers.filter((h) => h !== handler);
    };
  }

  onDisconnect(handler: ConnectionHandler): () => void {
    this.disconnectHandlers.push(handler);
    return () => {
      this.disconnectHandlers = this.disconnectHandlers.filter((h) => h !== handler);
    };
  }

  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.push(handler);
    return () => {
      this.errorHandlers = this.errorHandlers.filter((h) => h !== handler);
    };
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}