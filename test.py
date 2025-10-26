import subprocess
script = '''
tell application "Numbers"
    activate
    make new document
    tell document 1
        tell sheet 1
            tell table 1
                set value of cell "A1" to 10.2
                set value of cell "A2" to 15.1
                set value of cell "A3" to (get value of cell "A1") + (get value of cell "A2")
                return value of cell "A3"
            end tell
        end tell
    end tell
end tell
'''
result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
a3_value = result.stdout.strip()
print(a3_value)