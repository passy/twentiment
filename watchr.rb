watch ('(tests/.*\.py)|(twentiment/.*\.py)') {|md| code_changed "#{md[0]}"}

def code_changed(file)
    if file =~ /__pycache__/
        return
    end

    print "Change detected in #{file} …"
    result = `python -m unittest`
end
