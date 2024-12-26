import sys
from .main import main
from .gui import launch_gui

if __name__ == '__main__':
    if '--gui' in sys.argv:
        sys.argv.remove('--gui')
        launch_gui()
    else:
        main()
