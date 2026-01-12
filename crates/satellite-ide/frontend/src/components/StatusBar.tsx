import React from 'react';
import { GitBranch, Cpu, AlertCircle, CheckCircle } from 'lucide-react';
import './StatusBar.css';

export interface StatusBarItem {
    id: string;
    position: 'left' | 'right';
    content: React.ReactNode;
    onClick?: () => void;
}

interface StatusBarProps {
    items?: StatusBarItem[];
    line?: number;
    column?: number;
    language?: string;
    encoding?: string;
    branch?: string;
    gpuStatus?: 'ready' | 'busy' | 'unavailable';
    message?: string;
}

export function StatusBar({
    items = [],
    line = 1,
    column = 1,
    language = 'satellite',
    encoding = 'UTF-8',
    branch = 'main',
    gpuStatus = 'unavailable',
    message = 'Ready',
}: StatusBarProps) {
    const leftItems = items.filter(i => i.position === 'left');
    const rightItems = items.filter(i => i.position === 'right');

    const gpuIcon = {
        ready: <CheckCircle size={14} className="gpu-ready" />,
        busy: <Cpu size={14} className="gpu-busy" />,
        unavailable: <AlertCircle size={14} className="gpu-unavailable" />,
    }[gpuStatus];

    return (
        <div className="status-bar">
            <div className="status-bar-left">
                <div className="status-item branch">
                    <GitBranch size={14} />
                    <span>{branch}</span>
                </div>
                <div className="status-item message">
                    {message}
                </div>
                {leftItems.map(item => (
                    <div
                        key={item.id}
                        className="status-item"
                        onClick={item.onClick}
                    >
                        {item.content}
                    </div>
                ))}
            </div>

            <div className="status-bar-right">
                {rightItems.map(item => (
                    <div
                        key={item.id}
                        className="status-item"
                        onClick={item.onClick}
                    >
                        {item.content}
                    </div>
                ))}
                <div className="status-item">
                    Ln {line}, Col {column}
                </div>
                <div className="status-item">
                    {language}
                </div>
                <div className="status-item">
                    {encoding}
                </div>
                <div className="status-item gpu-status" title={`GPU: ${gpuStatus}`}>
                    {gpuIcon}
                    <span>GPU</span>
                </div>
            </div>
        </div>
    );
}

export default StatusBar;
