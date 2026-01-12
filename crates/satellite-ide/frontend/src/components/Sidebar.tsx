import React from 'react';
import './Sidebar.css';

interface SidebarProps {
    position: 'left' | 'right';
    visible: boolean;
    width?: number;
    children: React.ReactNode;
}

export function Sidebar({ position, visible, width = 240, children }: SidebarProps) {
    if (!visible) return null;

    return (
        <div
            className={`sidebar sidebar-${position}`}
            style={{ width }}
        >
            {children}
        </div>
    );
}

export default Sidebar;
