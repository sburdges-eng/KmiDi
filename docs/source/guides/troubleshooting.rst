Troubleshooting Guide
=====================

Common Issues
------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: bash

   # Verify Python path
   python3 -c "import sys; print(sys.path)"
   
   # Run import verification
   python3 scripts/verify_imports.py

Build Issues
~~~~~~~~~~~

If C++ build fails:

.. code-block:: bash

   # Check CMake configuration
   python3 scripts/verify_build.py
   
   # Rebuild from scratch
   rm -rf build
   ./scripts/setup_build.sh

Performance Issues
~~~~~~~~~~~~~~~~~

- Profile imports: ``python3 scripts/profile_imports.py``
- Check C++ library is built: ``ls build/src_penta-core/libpenta_core.a``
- Verify Python bindings: ``python3 scripts/test_cpp_bridge.py``

Getting Help
------------

- Check existing issues on GitHub
- Review documentation
- Run verification scripts
- Check logs for detailed error messages
