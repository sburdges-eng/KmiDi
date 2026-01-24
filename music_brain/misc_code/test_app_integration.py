"""
Integration Tests for DAiW Desktop App

End-to-end tests for the complete application flow.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAppLaunchFlow:
    """Test the complete app launch flow"""
    
    @patch('launcher.find_free_port')
    @patch('launcher.run_streamlit')
    @patch('launcher.wait_for_server')
    @patch('launcher.start_webview')
    def test_complete_launch_flow(self, mock_webview, mock_wait, mock_run, mock_port):
        """Test the complete launch flow from start to finish"""
        import launcher
        
        # Setup mocks
        mock_port.return_value = 8501
        mock_process = MagicMock()
        mock_run.return_value = mock_process
        mock_wait.return_value = True
        
        # Run main
        launcher.main()
        
        # Verify flow
        mock_port.assert_called_once()
        mock_run.assert_called_once_with(8501)
        mock_wait.assert_called_once()
        mock_webview.assert_called_once()
        mock_process.terminate.assert_called()
        mock_process.wait.assert_called()
    
    def test_port_generation(self):
        """Test that port generation works correctly"""
        import launcher
        
        port = launcher.find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


class TestAppDependencies:
    """Test app dependencies and requirements"""
    
    def test_streamlit_available(self):
        """Test that Streamlit is available"""
        try:
            import streamlit
            assert streamlit is not None
        except ImportError:
            pytest.skip("Streamlit not installed")
    
    def test_pywebview_optional(self):
        """Test that pywebview is optional (fallback to browser)"""
        try:
            import webview
            assert webview is not None
        except ImportError:
            # This is OK - app falls back to browser
            assert True
    
    def test_music_brain_modules(self):
        """Test that music_brain modules are importable"""
        try:
            from music_brain import ParrotVocalSynthesizer
            from music_brain import AudioAnalyzer
            from music_brain import HarmonyGenerator
            assert True
        except ImportError as e:
            pytest.skip(f"music_brain modules not available: {e}")


class TestAppConfiguration:
    """Test app configuration and settings"""
    
    def test_app_title_constant(self):
        """Test that app title constant exists"""
        import launcher
        assert hasattr(launcher, 'APP_TITLE')
        assert isinstance(launcher.APP_TITLE, str)
        assert len(launcher.APP_TITLE) > 0
    
    def test_streamlit_script_constant(self):
        """Test that Streamlit script constant exists"""
        import launcher
        assert hasattr(launcher, 'STREAMLIT_SCRIPT')
        assert launcher.STREAMLIT_SCRIPT == "app.py"
    
    def test_app_file_exists(self):
        """Test that app.py exists"""
        app_path = Path(__file__).parent.parent / "app.py"
        assert app_path.exists(), "app.py should exist"


class TestErrorHandling:
    """Test error handling in the app"""
    
    @patch('launcher.run_streamlit')
    @patch('launcher.wait_for_server')
    def test_server_startup_failure_handling(self, mock_wait, mock_run):
        """Test handling when server fails to start"""
        import launcher
        
        mock_process = MagicMock()
        mock_run.return_value = mock_process
        mock_wait.return_value = False  # Server failed to start
        
        with pytest.raises(RuntimeError, match="Streamlit server failed to start"):
            launcher.main()
        
        # Process should be terminated
        mock_process.terminate.assert_called()
    
    @patch('launcher.run_streamlit')
    @patch('launcher.wait_for_server')
    @patch('launcher.start_webview')
    def test_cleanup_on_exception(self, mock_webview, mock_wait, mock_run):
        """Test that process is cleaned up even if webview fails"""
        import launcher
        
        mock_process = MagicMock()
        mock_run.return_value = mock_process
        mock_wait.return_value = True
        mock_webview.side_effect = Exception("Webview error")
        
        with pytest.raises(Exception):
            launcher.main()
        
        # Process should still be terminated
        mock_process.terminate.assert_called()


class TestStreamlitIntegration:
    """Test Streamlit app integration"""
    
    def test_streamlit_command_generation(self):
        """Test that Streamlit command is generated correctly"""
        import launcher
        import sys
        
        port = 8501
        process = launcher.run_streamlit(port)
        
        # Process should be created
        assert process is not None
    
    @patch('subprocess.Popen')
    def test_streamlit_command_args(self, mock_popen):
        """Test that Streamlit command has correct arguments"""
        import launcher
        import sys
        
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        port = 8501
        launcher.run_streamlit(port)
        
        # Check command was called
        assert mock_popen.called
        call_args = mock_popen.call_args[0][0]
        
        # Check for important arguments
        assert "--server.port" in call_args
        assert str(port) in call_args
        assert "--server.headless" in call_args
        assert "true" in call_args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

