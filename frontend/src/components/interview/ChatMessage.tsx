// frontend/src/components/interview/ChatMessage.tsx

'use client';

import { cn } from '@/lib/utils';
import { User, Bot, Volume2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ChatMessageProps {
  role: 'interviewer' | 'candidate' | 'system';
  content: string;
  timestamp?: Date;
  isFollowUp?: boolean;
  audioBase64?: string;
  onPlayAudio?: () => void;
}

export function ChatMessage({
  role,
  content,
  timestamp,
  isFollowUp,
  audioBase64,
  onPlayAudio,
}: ChatMessageProps) {
  const isInterviewer = role === 'interviewer';
  const isSystem = role === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center my-4">
        <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
          {content}
        </span>
      </div>
    );
  }

  const playAudio = () => {
    if (audioBase64) {
      const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
      audio.play().catch(console.error);
      onPlayAudio?.();
    }
  };

  return (
    <div
      className={cn(
        'flex gap-3 mb-4',
        isInterviewer ? 'justify-start' : 'justify-end'
      )}
    >
      {isInterviewer && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center">
          <Bot className="w-5 h-5 text-orange-600" />
        </div>
      )}

      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-3',
          isInterviewer
            ? 'bg-gray-100 text-gray-900 rounded-tl-none'
            : 'bg-gray-900 text-white rounded-tr-none'
        )}
      >
        {isFollowUp && isInterviewer && (
          <span className="text-xs text-orange-600 font-medium block mb-1">
            🔄 Follow-up Question
          </span>
        )}
        
        <p className="text-sm whitespace-pre-wrap">{content}</p>
        
        <div className="flex items-center justify-between mt-2">
          {timestamp && (
            <span
              className={cn(
                'text-xs',
                isInterviewer ? 'text-gray-500' : 'text-gray-400'
              )}
            >
              {timestamp.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          )}
          
          {audioBase64 && isInterviewer && (
            <Button
              variant="ghost"
              size="sm"
              onClick={playAudio}
              className="h-6 px-2 text-xs"
            >
              <Volume2 className="w-3 h-3 mr-1" />
              Play
            </Button>
          )}
        </div>
      </div>

      {!isInterviewer && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-900 flex items-center justify-center">
          <User className="w-5 h-5 text-white" />
        </div>
      )}
    </div>
  );
}