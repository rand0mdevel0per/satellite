import { useState, useEffect } from 'react';
import {
    ChevronRight,
    ChevronDown,
    File,
    Folder,
    FolderOpen,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import './FileExplorer.css';

interface FileNode {
    name: string;
    path: string;
    isDir: boolean;
    children?: FileNode[];
}

interface FileExplorerProps {
    onFileSelect: (path: string) => void;
    rootPath?: string;
}

export function FileExplorer({ onFileSelect, rootPath }: FileExplorerProps) {
    const [tree, setTree] = useState<FileNode[]>([]);
    const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
    const [selectedPath, setSelectedPath] = useState<string | null>(null);

    useEffect(() => {
        loadDirectory(rootPath || '.');
    }, [rootPath]);

    const loadDirectory = async (path: string) => {
        try {
            const entries = await invoke<FileNode[]>('list_directory', { path });
            setTree(entries);
        } catch (err) {
            console.error('Failed to load directory:', err);
            // Fallback mock data for development
            setTree([
                { name: 'src', path: './src', isDir: true },
                { name: 'main.sat', path: './main.sat', isDir: false },
                { name: 'config.json', path: './config.json', isDir: false },
            ]);
        }
    };

    const toggleExpand = async (node: FileNode) => {
        const newExpanded = new Set(expandedPaths);
        if (newExpanded.has(node.path)) {
            newExpanded.delete(node.path);
        } else {
            newExpanded.add(node.path);
            // Load children if not already loaded
            if (!node.children) {
                try {
                    const children = await invoke<FileNode[]>('list_directory', { path: node.path });
                    node.children = children;
                    setTree([...tree]);
                } catch (err) {
                    console.error('Failed to load:', err);
                }
            }
        }
        setExpandedPaths(newExpanded);
    };

    const handleClick = (node: FileNode) => {
        setSelectedPath(node.path);
        if (node.isDir) {
            toggleExpand(node);
        } else {
            onFileSelect(node.path);
        }
    };

    const renderNode = (node: FileNode, depth: number = 0) => {
        const isExpanded = expandedPaths.has(node.path);
        const isSelected = selectedPath === node.path;

        return (
            <div key={node.path}>
                <div
                    className={`file-node ${isSelected ? 'selected' : ''}`}
                    style={{ paddingLeft: `${depth * 16 + 8}px` }}
                    onClick={() => handleClick(node)}
                >
                    {node.isDir ? (
                        <>
                            {isExpanded ? (
                                <ChevronDown size={16} className="icon-chevron" />
                            ) : (
                                <ChevronRight size={16} className="icon-chevron" />
                            )}
                            {isExpanded ? (
                                <FolderOpen size={16} className="icon-folder" />
                            ) : (
                                <Folder size={16} className="icon-folder" />
                            )}
                        </>
                    ) : (
                        <>
                            <span className="icon-spacer" />
                            <File size={16} className="icon-file" />
                        </>
                    )}
                    <span className="file-name">{node.name}</span>
                </div>
                {node.isDir && isExpanded && node.children && (
                    <div className="file-children">
                        {node.children.map(child => renderNode(child, depth + 1))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="file-explorer">
            <div className="file-explorer-header">
                <span>EXPLORER</span>
            </div>
            <div className="file-tree">
                {tree.map(node => renderNode(node))}
            </div>
        </div>
    );
}

export default FileExplorer;
