import subprocess
applescript = '''
tell application "Numbers"
    activate
    make new document
    tell document 1
        tell sheet 1
            tell table 1
                set value of cell "A1" to 10
                set value of cell "A2" to 15
                set value of cell "A3" to (get value of cell "A1") + (get value of cell "A2")
            end tell
        end tell
    end tell
end tell
'''
subprocess.run(['osascript', '-e', applescript])