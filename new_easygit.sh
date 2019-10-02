#!/bin/bash
# easygit Bash script, by bassandaruwan
# Easy handling of git pull, add, commit and push
# Usage:
#     1. bash easygit.sh -h
#         Get help
#     1. bash easygit.sh
#         Check git status
#     2. bash easygit.sh 'comment'
#         Git add, commit (with comment) and push
#     3. bash easygit.sh -c 'comment'
#         Only git add . and commit with 'comment'
#     4. bash easygit.sh -p
#         Only git push

echo
echo ===== Easy-Git =====
echo

if [ $# -gt 0 ]; then
    case "$1" in
    -h | --help)
        echo "$package - easy git use"
        echo " "
        echo "options:"
        echo "-h, --help    show brief help"
        echo "-c,           git commit message"
        exit 0
        ;;
    -c)
        shift
        echo "*** only git add & commit"
        echo 

        if [ $# -eq 1 ]; then
            echo ">>" Commit message: "$1"
            echo

            echo
            echo ">>" Checking git status
            echo
            # git status

            echo
            echo ">>" Git add
            echo
            # git add .

            echo
            echo ">>" Git commit
            echo
            # git commit -m "$1"


        else
            echo "Commit message required"
        fi
        exit 0
        ;;
    -p)
        echo "*** only git push"
        echo 

        if [ $# -eq 0 ]; then
            echo
            echo ">>" Git push
            echo
            # git push

        fi
        exit 0
        ;;
    *)
        echo "Not valid argument"
        exit 0
        ;;
    esac
fi

if [ $# -eq 0 ]; then
    echo
    echo ">>" Checking git status
    echo
    git status
    # echo ERROR: Need commit message as argument
    exit
fi

if [ $# -eq 1 ]; then
    echo ">>" Commit message: "$1"
    echo
    echo
    echo ">>" Git pull
    echo
    # git pull

    echo
    echo ">>" Checking git status
    echo
    # git status

    echo
    echo ">>" Git add
    echo
    # git add .

    echo
    echo ">>" Git commit
    echo
    # git commit -m "$1"

    echo
    echo ">>" Git push
    echo
    # git push

    echo
    echo ">>" Checking git status
    echo
    # git status
fi
