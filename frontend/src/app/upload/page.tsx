// frontend/src/app/upload/page.tsx

import { ResumeUploader } from '@/components/upload/ResumeUploader';

export default function UploadPage() {
  return (
    <div className="container mx-auto max-w-2xl px-4 py-12">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Start Your Mock Interview</h1>
        <p className="text-gray-600">
          Upload your resume and configure your interview settings
        </p>
      </div>

      <ResumeUploader />
    </div>
  );
}