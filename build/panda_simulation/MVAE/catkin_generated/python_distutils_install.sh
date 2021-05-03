#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/cristian/Scrivania/MAIF/src/panda_simulation/MVAE"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/cristian/Scrivania/MAIF/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/cristian/Scrivania/MAIF/install/lib/python2.7/dist-packages:/home/cristian/Scrivania/MAIF/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/cristian/Scrivania/MAIF/build" \
    "/usr/bin/python2" \
    "/home/cristian/Scrivania/MAIF/src/panda_simulation/MVAE/setup.py" \
     \
    build --build-base "/home/cristian/Scrivania/MAIF/build/panda_simulation/MVAE" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/cristian/Scrivania/MAIF/install" --install-scripts="/home/cristian/Scrivania/MAIF/install/bin"
