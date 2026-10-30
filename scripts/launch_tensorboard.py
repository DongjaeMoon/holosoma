"""Use built-in TensorBoard plugins, avoiding unrelated Open3D dependencies."""
import sys
from tensorboard import default, program

board = program.TensorBoard(plugins=default.get_static_plugins())
board.configure(argv=['tensorboard', *sys.argv[1:]])
board.main()
