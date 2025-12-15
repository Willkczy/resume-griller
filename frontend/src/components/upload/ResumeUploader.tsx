// frontend/src/components/upload/ResumeUploader.tsx

'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Upload, FileText, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { uploadResume, createSession } from '@/lib/api';
import type { InterviewMode, ResumeUploadResponse } from '@/types';
import { cn } from '@/lib/utils';

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export function ResumeUploader() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [uploadResult, setUploadResult] = useState<ResumeUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedMode, setSelectedMode] = useState<InterviewMode>('mixed');
  const [numQuestions, setNumQuestions] = useState(5);
  const [isStarting, setIsStarting] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.type === 'application/pdf' || droppedFile.name.endsWith('.txt'))) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please upload a PDF or TXT file');
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  }, []);

  const handleUpload = async () => {
    if (!file) return;

    setUploadStatus('uploading');
    setError(null);

    try {
      const result = await uploadResume(file);
      setUploadResult(result);
      setUploadStatus('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploadStatus('error');
    }
  };

  const handleStartInterview = async () => {
    if (!uploadResult) return;

    setIsStarting(true);
    setError(null);

    try {
      const response = await createSession({
        resume_id: uploadResult.resume_id,
        mode: selectedMode,
        num_questions: numQuestions,
      });

      const sessionId = response.metadata?.session_id;
      if (sessionId) {
        router.push(`/interview/${sessionId}`);
      } else {
        throw new Error('No session ID returned');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start interview');
      setIsStarting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card>
        <CardHeader>
          <CardTitle>Upload Your Resume</CardTitle>
          <CardDescription>
            Upload your resume in PDF or TXT format to start your mock interview
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              'border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer',
              isDragging
                ? 'border-gray-900 bg-gray-50'
                : 'border-gray-300 hover:border-gray-400',
              uploadStatus === 'success' && 'border-green-500 bg-green-50'
            )}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept=".pdf,.txt"
              onChange={handleFileSelect}
              className="hidden"
            />

            {uploadStatus === 'uploading' ? (
              <div className="flex flex-col items-center">
                <Loader2 className="h-12 w-12 text-gray-400 animate-spin" />
                <p className="mt-4 text-sm text-gray-600">Uploading and processing...</p>
              </div>
            ) : uploadStatus === 'success' ? (
              <div className="flex flex-col items-center">
                <CheckCircle className="h-12 w-12 text-green-500" />
                <p className="mt-4 text-sm font-medium text-green-700">
                  Resume uploaded successfully!
                </p>
                <p className="text-sm text-gray-600">
                  {uploadResult?.chunks_created} sections extracted
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                {file ? (
                  <>
                    <FileText className="h-12 w-12 text-gray-400" />
                    <p className="mt-4 text-sm font-medium">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </>
                ) : (
                  <>
                    <Upload className="h-12 w-12 text-gray-400" />
                    <p className="mt-4 text-sm text-gray-600">
                      <span className="font-medium">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">PDF or TXT (max 10MB)</p>
                  </>
                )}
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 flex items-center gap-2 text-red-600">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {file && uploadStatus === 'idle' && (
            <Button onClick={handleUpload} className="mt-4 w-full">
              Upload Resume
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Interview Settings */}
      {uploadStatus === 'success' && uploadResult && (
        <Card>
          <CardHeader>
            <CardTitle>Interview Settings</CardTitle>
            <CardDescription>
              Configure your mock interview preferences
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Mode Selection */}
            <div>
              <label className="text-sm font-medium mb-3 block">Interview Mode</label>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { value: 'hr', label: 'HR', desc: 'Behavioral questions' },
                  { value: 'tech', label: 'Technical', desc: 'Technical deep-dive' },
                  { value: 'mixed', label: 'Mixed', desc: 'Both types' },
                ].map((mode) => (
                  <button
                    key={mode.value}
                    onClick={() => setSelectedMode(mode.value as InterviewMode)}
                    className={cn(
                      'p-4 rounded-lg border-2 text-left transition-colors',
                      selectedMode === mode.value
                        ? 'border-gray-900 bg-gray-50'
                        : 'border-gray-200 hover:border-gray-300'
                    )}
                  >
                    <div className="font-medium">{mode.label}</div>
                    <div className="text-xs text-gray-500">{mode.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Number of Questions */}
            <div>
              <label className="text-sm font-medium mb-3 block">
                Number of Questions: {numQuestions}
              </label>
              <input
                type="range"
                min="3"
                max="10"
                value={numQuestions}
                onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>3 (Quick)</span>
                <span>10 (Thorough)</span>
              </div>
            </div>

            {/* Resume Info */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-sm mb-2">Resume Summary</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <p>📄 {uploadResult.filename}</p>
                <p>📊 {uploadResult.chunks_created} sections extracted</p>
                <p>🏷️ Sections: {uploadResult.sections.join(', ')}</p>
              </div>
            </div>

            <Button
              onClick={handleStartInterview}
              disabled={isStarting}
              className="w-full"
              size="lg"
            >
              {isStarting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Starting Interview...
                </>
              ) : (
                '🔥 Start Interview'
              )}
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}