# Partial Prompt Taxonomy

| ID | Category | Case | Example (prompt → image) | Minimal pair | Anchor | Expected penalty | Pilot |
|---|---|---|---|---|---|---|---|
| A1 | A · Forced completion | Unspecified attribute | *an apple* → a green apple | A red apple; ratings should not differ | Underspecification | none | S01 |
| A2 | A · Forced completion | Background / scene | *a chair on the grass* → sky, sun, cloud, tree | Same scene with a bare sky | Underspecification | none | S02 |
| A3 | A · Forced completion | Viewpoint / framing | *a cup of coffee* → seen from directly above | Same cup, side view | Underspecification | none | S03 |
| A′1 | A' · Atypical completion | Atypical category member | *a bird* → a penguin | A sparrow | Prototype theory (Rosch) | graded by typicality | new |
| A′2 | A' · Atypical completion | Atypical context | *a polar bear* → in a desert | On sea ice | Scene schema, typicality | low to medium | new |
| A′3 | A' · Atypical completion | Counterfactual attribute | *an apple* → a blue apple | A red apple | Prototype theory | medium | new |
| B1 | B · Implied content | Implied presence, realized | *a birthday cake* → lit candles | Same cake without candles (B2) | Scripts, bridging inference | none, possible bonus | S04 |
| B2 | B · Implied content | Implied presence, missing | *a birthday cake* → no candles | Same cake with candles (B1) | Scripts, bridging inference | low to medium | new |
| B3 | B · Implied content | Implied absence, violated | *a lonely tree in a field* → a second tree | Same image under “a tree in a field” | Implicit (lexical) negation | high | new |
| C1 | C · Optional addition | Typical context object | *a laptop on a desk* → a desk lamp and a mug | The laptop alone | Scene schema | low | S05 |
| C2 | C · Optional addition | Competing object | *an apple* → an apple and a pear | The apple alone | Gricean quantity (exhaustivity) | medium, high disagreement | S06, S11 |
| C3 | C · Optional addition | Extra instance | *a chair on the grass* → two chairs | One chair | Scalar implicature (cardinality) | high | S07 |
| C4 | C · Optional addition | Extra instance × relation | *a cat to the left of a dog* → two dogs, the cat between them | One dog, to the right of the cat | Uniqueness presupposition | medium to high | new |
| C5 | C · Optional addition | Added relation | *a cat and a box* → the cat inside the box | The cat next to the box | Relation hallucination | low to medium | S08 |
| C6 | C · Optional addition | Modified subject | *a dog* → a dog wearing sunglasses | The same dog without them | Category boundaries, object state | graded: accessory below state change | new |
| C7 | C · Optional addition | Semantic violation | *an apple on a table* → a traffic cone | The apple alone | Scene grammar, semantic (Võ) | high | S09 |
| C8 | C · Optional addition | Syntactic violation | *an apple on a table* → a mug floating above the table | The mug resting on the table | Scene grammar, syntactic (Võ) | high, may be read as quality | new |
| C9 | C · Optional addition | Added animate agent | *a park bench* → a person sitting on it | A bag of the same size on the bench | Animacy captures attention | medium, above C1 | new |
| D1 | D · Salience | Subject too small | *a sailboat* → a tiny boat on the horizon | The boat filling the frame | Visual salience | medium | S10 |
| D2 | D · Salience | Inverted salience hierarchy | *a man with an umbrella* → a giant umbrella, a tiny man | Normal proportions; or one image under “a cat and a dog” and “a dog and a cat” | Order of mention and salience (Clarke et al., 2015) | low to medium, open | new |
| E1 | E · Control | Attention check | *a banana* → an apple | A banana | Annotator quality | rating 1 expected | S12 |
| G1 | G · Generator-specific | Attribute spill-over | *a red apple* → a red table and a red wall | The red apple in a neutral room | Semantic leakage | low to medium | new |
| G2 | G · Generator-specific | Double sense | *a bat* → the animal and a baseball bat | The animal only | Homonym duplication (Rassin et al., 2022) | medium | new |
| G3 | G · Generator-specific | Depicted instance | *an apple* → a poster of an apple on a wall | A real apple | Depiction vs instance | high | new |
| G4 | G · Generator-specific | Generator artefacts | *a chair on the grass* → pseudo-text in a corner | The clean image | Alignment vs quality | low on alignment | new |

## Cross-cutting axes

| Axis | Levels | Crossed with | What it tests |
|---|---|---|---|
| Exhaustivity | “apple” · “an apple” · “a single apple” · “an apple among other fruit” | C2, C3 | If cancelling the implicature removes the penalty, the effect is pragmatic rather than semantic. |
| Prompt specificity | “an apple” · “a red apple on a wooden table” | C1–C9 | Whether detailed prompts are read as exhaustive. |
| Prompt register | Sentence · tag list (“apple, wooden table, 4k”) | C3, D2 | Tag lists have no article and no head noun, so no cardinality implicature and no syntactic hierarchy. |
| Prompt type | Object prompt (“a mug”) · scene prompt (“a kitchen”) | C1, C7, C8 | How much unmentioned content a scene schema licenses. |
| Animacy | Inanimate · animate addition of matched size | C1–C3, C9 | Whether animate additions are penalized more. |
| Dose | 0 · 1 · 3 · 10 added objects; small · large area | C1, C2 | A penalty curve instead of a binary effect. |
| Reference zero | Subject alone on a neutral background | A, C | A floor for measuring the effect of any completion. |
