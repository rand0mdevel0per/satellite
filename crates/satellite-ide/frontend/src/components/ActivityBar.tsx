import {
    Files,
    GitBranch,
    Search,
    Terminal,
    ListTodo,
    Settings,
    LucideIcon,
} from 'lucide-react';
import './ActivityBar.css';

export interface ActivityItem {
    id: string;
    icon: LucideIcon;
    label: string;
    position: 'top' | 'bottom';
}

interface ActivityBarProps {
    items: ActivityItem[];
    activeId: string | null;
    onItemClick: (id: string) => void;
}

// Default activity items
export const defaultActivityItems: ActivityItem[] = [
    { id: 'files', icon: Files, label: 'Explorer', position: 'top' },
    { id: 'git', icon: GitBranch, label: 'Source Control', position: 'top' },
    { id: 'search', icon: Search, label: 'Search', position: 'top' },
    { id: 'terminal', icon: Terminal, label: 'Terminal', position: 'bottom' },
    { id: 'tasks', icon: ListTodo, label: 'Tasks', position: 'bottom' },
    { id: 'settings', icon: Settings, label: 'Settings', position: 'bottom' },
];

export function ActivityBar({ items, activeId, onItemClick }: ActivityBarProps) {
    const topItems = items.filter(i => i.position === 'top');
    const bottomItems = items.filter(i => i.position === 'bottom');

    return (
        <div className="activity-bar">
            <div className="activity-bar-top">
                {topItems.map(item => (
                    <button
                        key={item.id}
                        className={`activity-item ${activeId === item.id ? 'active' : ''}`}
                        onClick={() => onItemClick(item.id)}
                        title={item.label}
                    >
                        <item.icon size={24} />
                    </button>
                ))}
            </div>
            <div className="activity-bar-bottom">
                {bottomItems.map(item => (
                    <button
                        key={item.id}
                        className={`activity-item ${activeId === item.id ? 'active' : ''}`}
                        onClick={() => onItemClick(item.id)}
                        title={item.label}
                    >
                        <item.icon size={24} />
                    </button>
                ))}
            </div>
        </div>
    );
}

export default ActivityBar;
