for dir in kb{a..z} kc{a..u}; do
  if [ -d "$dir" ]; then
    for year in {1955..2099}; do
      year_path="$dir/$year"
      if [ -d "$year_path" ]; then
        count=$(find "$year_path" -type f -name "*daily*" | wc -l)
        if [ "$count" -lt 12 ]; then
          echo "MISSING: $year_path contains $count daily files"
        fi
      fi
    done
  fi
done
