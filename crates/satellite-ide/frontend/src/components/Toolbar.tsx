import {
    Play,
    Square,
    Bug,
    Save,
    FolderOpen,
    Settings,
    HelpCircle,
} from 'lucide-react';
import './Toolbar.css';

interface ToolbarProps {
    onSolve: () => void;
    onStop?: () => void;
    onAnalyze?: () => void;
    onSave?: () => void;
    onOpen?: () => void;
    onSettings?: () => void;
    solving?: boolean;
}

export function Toolbar({
    onSolve,
    onStop,
    onAnalyze,
    onSave,
    onOpen,
    onSettings,
    solving = false,
}: ToolbarProps) {
    return (
        <div className="toolbar">
            <div className="toolbar-left">
                <div className="toolbar-group">
                    <button className="toolbar-btn" onClick={onOpen} title="Open File">
                        <FolderOpen size={16} />
                    </button>
                    <button className="toolbar-btn" onClick={onSave} title="Save">
                        <Save size={16} />
                    </button>
                </div>

                <div className="toolbar-separator" />

                <div className="toolbar-group">
                    {solving ? (
                        <button className="toolbar-btn stop" onClick={onStop} title="Stop">
                            <Square size={16} />
                            <span>Stop</span>
                        </button>
                    ) : (
                        <button className="toolbar-btn run" onClick={onSolve} title="Solve">
                            <Play size={16} />
                            <span>Solve</span>
                        </button>
                    )}
                    <button className="toolbar-btn" onClick={onAnalyze} title="Analyze">
                        <Bug size={16} />
                        <span>Analyze</span>
                    </button>
                </div>
            </div>

            <div className="toolbar-center">
                <span className="toolbar-title">Satellite IDE</span>
            </div>

            <div className="toolbar-right">
                <button className="toolbar-btn" onClick={onSettings} title="Settings">
                    <Settings size={16} />
                </button>
                <button className="toolbar-btn" title="Help">
                    <HelpCircle size={16} />
                </button>
            </div>
        </div>
    );
}

export default Toolbar;
