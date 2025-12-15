// frontend/src/components/layout/Header.tsx

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Flame } from 'lucide-react';

export function Header() {
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Home' },
    { href: '/upload', label: 'Start Interview' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-gray-200 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="container mx-auto flex h-14 max-w-screen-xl items-center px-4">
        <Link href="/" className="flex items-center space-x-2">
          <Flame className="h-6 w-6 text-orange-500" />
          <span className="font-bold text-xl">Resume Griller</span>
        </Link>

        <nav className="ml-auto flex items-center space-x-6">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'text-sm font-medium transition-colors hover:text-gray-900',
                pathname === item.href
                  ? 'text-gray-900'
                  : 'text-gray-500'
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}