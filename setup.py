from setuptools import setup

setup(name='motion_primitives_py', version='0.0.1', packages=['motion_primitives_py'], python_requires='>3.6',
      install_requires=['ujson',
                        'sympy',
                        'matplotlib',
                        'numpy',
                        'pytest',
                        'py_opt_control @ git+ssh://git@github.com/jpaulos/opt_control#egg=py_opt_control&subdirectory=python',
                        'reeds_shepp @ git+ssh://git@github.com/ghliu/pyReedsShepp#egg=reeds_shepp',
                        'sobol_seq @ git+ssh://git@github.com/naught101/sobol_seq@v0.2.0#egg=sobol_seq'
                        ]
      )
