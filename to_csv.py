import csv

reviews=[]
labels=[]

with open("data/Compiled_Reviews.txt") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews.append(fields[0])
        labels.append(fields[2])

# Remove empty review
reviews.pop(33402)
labels.pop(33402)

rows = zip(reviews, labels)
with open('data/reviews.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Reviews', 'label'])
    writer.writerows(rows)
