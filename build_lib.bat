@echo off
setlocal enabledelayedexpansion

REM Step 1: Get version from uv
for /f "delims=" %%v in ('uv version --short') do set "version=%%v"

REM Step 2: Extract major.minor
for /f "tokens=1,2 delims=." %%a in ("!version!") do (
    set "major=%%a"
    set "minor=%%b"
)

REM Step 3: Get SVN revision
svn update
for /f %%r in ('svn info --show-item last-changed-revision') do set "revision=%%r"

REM Step 4: Build full and short version strings
set "short_version=%major%.%minor%"
set "full_version=%major%.%minor%.%revision%"

REM Step 5: Run the command with the new version
uv version --no-index --find-links=file:"\\nas.sg.gbs.pro\AreaFinanza\Share\PROGETTI ML\Python\python3_11\python packages" %full_version%

REM Step 6: Build the package into a wheel
uv build --no-index --find-links=file:"\\nas.sg.gbs.pro\AreaFinanza\Share\PROGETTI ML\Python\python3_11\python packages" --wheel

REM Step 7: Extract project name from pyproject.toml file
for /f "tokens=1,* delims==" %%A in ('findstr "name =" pyproject.toml') do (
    set key=%%A
    set value=%%B
    set value=!value:"=!
    set value=!value: =!
    set project_name=!value!
    goto done
)

:done

REM Step 8: Copy wheel file into correct destination
copy "dist\%project_name%-%full_version%-py3-none-any.whl" "\\nas.sg.gbs.pro\AreaFinanza\Share\PROGETTI ML\Python\python3_11\python packages"

REM Step 9: Reset version in pyproject.toml file
uv version --no-index --find-links=file:"\\nas.sg.gbs.pro\AreaFinanza\Share\PROGETTI ML\Python\python3_11\python packages" %short_version%
