project(
    'pyswarm',
    'fortran', 'c',
    version: run_command(['python', 'bin/get_version.py'], check: true).stdout().strip()
)

subdir('src')
