// frontend/src/app/page.tsx

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Flame, Upload, MessageSquare, BarChart3, Mic } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="container mx-auto max-w-screen-xl px-4 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <div className="flex justify-center mb-6">
          <div className="relative">
            <Flame className="h-20 w-20 text-orange-500" />
            <div className="absolute -top-1 -right-1 w-6 h-6 bg-yellow-400 rounded-full animate-pulse" />
          </div>
        </div>
        
        <h1 className="text-5xl font-bold tracking-tight text-gray-900 mb-4">
          Resume Griller
        </h1>
        
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
          Get grilled by an AI interviewer that knows your resume inside out. 
          Practice with tough follow-up questions and improve your interview skills.
        </p>
        
        <div className="flex justify-center gap-4">
          <Link href="/upload">
            <Button size="lg" className="text-lg px-8">
              🔥 Start Interview
            </Button>
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
        <Card>
          <CardHeader>
            <Upload className="h-10 w-10 text-orange-500 mb-2" />
            <CardTitle>Upload Resume</CardTitle>
            <CardDescription>
              Upload your PDF or TXT resume for AI analysis
            </CardDescription>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader>
            <MessageSquare className="h-10 w-10 text-orange-500 mb-2" />
            <CardTitle>Smart Questions</CardTitle>
            <CardDescription>
              Get questions tailored specifically to your experience
            </CardDescription>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader>
            <Flame className="h-10 w-10 text-orange-500 mb-2" />
            <CardTitle>Get Grilled</CardTitle>
            <CardDescription>
              Vague answers trigger follow-up questions that dig deeper
            </CardDescription>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader>
            <Mic className="h-10 w-10 text-orange-500 mb-2" />
            <CardTitle>Voice Support</CardTitle>
            <CardDescription>
              Speak your answers naturally with voice recognition
            </CardDescription>
          </CardHeader>
        </Card>
      </div>

      {/* How It Works */}
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-8">How It Works</h2>
        
        <div className="space-y-6">
          <div className="flex gap-4 items-start">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center font-bold text-orange-600">
              1
            </div>
            <div>
              <h3 className="font-semibold text-lg">Upload Your Resume</h3>
              <p className="text-gray-600">
                Upload your resume in PDF or TXT format. Our AI will analyze it and extract key information about your skills, experience, and projects.
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center font-bold text-orange-600">
              2
            </div>
            <div>
              <h3 className="font-semibold text-lg">Choose Your Mode</h3>
              <p className="text-gray-600">
                Select HR mode for behavioral questions, Technical mode for deep-dive technical questions, or Mixed mode for a comprehensive interview.
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center font-bold text-orange-600">
              3
            </div>
            <div>
              <h3 className="font-semibold text-lg">Get Grilled</h3>
              <p className="text-gray-600">
                Answer questions via text or voice. Give vague answers? Expect follow-up questions that push you to be more specific. Just like a real interview!
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center font-bold text-orange-600">
              4
            </div>
            <div>
              <h3 className="font-semibold text-lg">Review & Improve</h3>
              <p className="text-gray-600">
                Get feedback on your answers and see where you can improve. Practice makes perfect!
              </p>
            </div>
          </div>
        </div>

        <div className="text-center mt-12">
          <Link href="/upload">
            <Button size="lg">
              Get Started →
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}