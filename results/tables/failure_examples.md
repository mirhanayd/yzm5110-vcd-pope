# Consistent changes between regular decoding and VCD (T=500), 3 seeds each

Broken = regular right in all 3 seeds, VCD wrong in all 3. Fixed = the reverse.

| split | change | image | question | label | regular (seed 1) | VCD (seed 1) |
|---|---|---|---|---|---|---|
| random | broken | COCO_val2014_000000303652.jpg | Is there a chair in the image? | no | No | Yes |
| random | broken | COCO_val2014_000000123570.jpg | Is there a carrot in the image? | no | There is no carrot or orange food seen in the image | Yes |
| random | broken | COCO_val2014_000000041180.jpg | Is there a boat in the image? | no | No | Yes |
| random | fixed | COCO_val2014_000000424792.jpg | Is there a chair in the image? | yes | No | Yes |
| random | fixed | COCO_val2014_000000304387.jpg | Is there a kite in the image? | yes | No | Yes |
| random | fixed | COCO_val2014_000000230175.jpg | Is there a cup in the image? | yes | No | Yes |
| popular | broken | COCO_val2014_000000406403.jpg | Is there a chair in the image? | no | No | Yes |
| popular | broken | COCO_val2014_000000506178.jpg | Is there a car in the image? | no | No | Yes |
| popular | broken | COCO_val2014_000000199940.jpg | Is there a person in the image? | no | No | Yes |
| popular | broken | COCO_val2014_000000388237.jpg | Is there a cup in the image? | no | No | Yes |
| popular | broken | COCO_val2014_000000141278.jpg | Is there a cup in the image? | no | No | Yes |
| popular | broken | COCO_val2014_000000190788.jpg | Is there a chair in the image? | no | No | Yes |
| popular | fixed | COCO_val2014_000000424792.jpg | Is there a chair in the image? | yes | No | Yes |
| popular | fixed | COCO_val2014_000000304387.jpg | Is there a kite in the image? | yes | No | Yes |
| popular | fixed | COCO_val2014_000000230175.jpg | Is there a cup in the image? | yes | No | Yes |
| adversarial | broken | COCO_val2014_000000374061.jpg | Is there a book in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000539251.jpg | Is there a bottle in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000506178.jpg | Is there a book in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000015338.jpg | Is there a traffic light in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000465418.jpg | Is there a handbag in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000199940.jpg | Is there a person in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000478736.jpg | Is there a truck in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000156282.jpg | Is there a car in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000041180.jpg | Is there a bicycle in the image? | no | No | Yes |
| adversarial | broken | COCO_val2014_000000190788.jpg | Is there a chair in the image? | no | No | Yes |
| adversarial | fixed | COCO_val2014_000000424792.jpg | Is there a chair in the image? | yes | No | Yes |
| adversarial | fixed | COCO_val2014_000000304387.jpg | Is there a kite in the image? | yes | No | Yes |
| adversarial | fixed | COCO_val2014_000000579277.jpg | Is there a bicycle in the image? | no | Yes | No |
| adversarial | fixed | COCO_val2014_000000230175.jpg | Is there a cup in the image? | yes | No | Yes |
