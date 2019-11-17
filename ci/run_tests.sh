#!/bin/bash -e

# Workaround for pytest-xdist (it collects different tests in the workers if PYTHONHASHSEED is not set)
# https://github.com/pytest-dev/pytest/issues/920
# https://github.com/pytest-dev/pytest/issues/1075
export PYTHONHASHSEED=$(python -c 'import random; print(random.randint(1, 4294967295))')

if [ -n "$LOCALE_OVERRIDE" ]; then
    export LC_ALL="$LOCALE_OVERRIDE"
    export LANG="$LOCALE_OVERRIDE"
    PANDAS_LOCALE=`python -c 'import pandas; pandas.get_option("display.encoding")'`
    if [[ "$LOCALE_OVERRIDE" != "$PANDAS_LOCALE" ]]; then
        echo "pandas could not detect the locale. System locale: $LOCALE_OVERRIDE, pandas detected: $PANDAS_LOCALE"
        # TODO Not really aborting the tests until https://github.com/pandas-dev/pandas/issues/23923 is fixed
        # exit 1
    fi
fi

if [[ "not network" == *"$PATTERN"* ]]; then
    export http_proxy=http://1.2.3.4 https_proxy=http://1.2.3.4;
fi

if [ "$COVERAGE" ]; then
    COVERAGE_FNAME="/tmp/test_coverage.xml"
    COVERAGE="-s --cov=pandas --cov-report=xml:$COVERAGE_FNAME"
fi

PYTEST_CMD="pytest -m \"$PATTERN\" -n auto --dist=loadfile -s --strict --durations=10 --junitxml=test-data.xml $TEST_ARGS $COVERAGE pandas"

# Travis does not have have an X server
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    DISPLAY=DISPLAY=:99.0
    PYTEST_CMD="xvfb-run -e /dev/stdout $PYTEST_CMD"
fi

echo $PYTEST_CMD
sh -c "$PYTEST_CMD"

if [[ "$COVERAGE" && $? == 0 && "$TRAVIS_BRANCH" == "master" ]]; then
    echo "uploading coverage"
    echo "bash <(curl -s https://codecov.io/bash) -Z -c -F $TYPE -f $COVERAGE_FNAME"
          bash <(curl -s https://codecov.io/bash) -Z -c -F $TYPE -f $COVERAGE_FNAME
fi
