# Strategy:

1) get lots of samples of literature
2) split them into ~5k chunks (chapters etc)
3) ask a very smart model to judge and label each chunk according to the rubric definitions

use the labeled data to populate a dataset of 1000 samples per {type x rubric x score} combination

For those combinations with fewer than 1000 samples:
synthetically generate more samples, using some existing samples as examples, and with the specific rubric x score definition

in the end, each literature type will have 1000 examples of score 1, 2, 3, 4, and 5 for each rubric - which ought to be enough to train a reward model to score arbitrary prose on each rubric

Train on https://huggingface.co/Qwen/WorldPM-72B

# Rubrics

## I. Structural Design & Development

*(Evaluates the overall organization, logical progression, and effective shaping of the content.)*

### A. Structural Coherence & Progression

* **5 (Exceptional)** – The structure is masterfully crafted, exhibiting flawless logical/thematic/narrative progression. All parts are intrinsically linked, contributing to a powerful and unified whole, perfectly suited to the work’s purpose and form.
* **4 (Accomplished)** – The structure is highly effective, with clear logical/thematic/narrative progression. Most parts are well-integrated, contributing to a cohesive work.
* **3 (Competent)** – The structure is generally clear and supports the content, though some areas might lack optimal flow or integration. Progression is mostly logical/thematic/narrative.
* **2 (Developing)** – Structural weaknesses are apparent; progression may be confusing, disjointed, or underdeveloped. Connections between parts are often unclear.
* **1 (Deficient)** – Lacks a discernible or effective structure; content is chaotic, randomly organized, or fails to develop coherently.

### B. Pacing & Rhythmic Management

* **5 (Exceptional)** – Pacing is expertly controlled, creating an ideal rhythm that maximizes engagement and impact appropriate to the form.
* **4 (Accomplished)** – Pacing is highly effective, varying appropriately to maintain interest and effectively deliver content/experience.
* **3 (Competent)** – Pacing is generally adequate, though some sections may feel disproportionately rushed, slow, or monotonous.
* **2 (Developing)** – Pacing is often uneven or ill-suited to the content/form, hindering engagement or clarity.
* **1 (Deficient)** – Pacing is detrimental, making the work feel stagnant, overwhelming, or difficult to follow.

### C. Focus & Central Idea / Purpose / Tension

* **5 (Exceptional)** – A clear and compelling central idea, purpose, narrative core, or thematic tension is established and masterfully developed throughout.
* **4 (Accomplished)** – A well-defined central idea/purpose/tension is maintained and effectively explored.
* **3 (Competent)** – A central idea/purpose/tension is generally present, though focus may occasionally waver or lack depth.
* **2 (Developing)** – The central idea/purpose/tension is unclear, poorly defined, or inconsistently addressed.
* **1 (Deficient)** – No clear central idea/purpose/tension; the work lacks focus or direction.

---

## II. Portrayal & Development of Subjects / Perspectives / Information

*(Evaluates the depth, clarity, authenticity, and development of characters, subjects, arguments, information, or perspectives.)*

### A. Depth, Nuance & Authenticity of Portrayal

* **5 (Exceptional)** – Subjects are presented with profound depth, nuance, and compelling authenticity, demonstrating sophisticated insight.
* **4 (Accomplished)** – Subjects are presented with considerable depth, clarity, and credibility.
* **3 (Competent)** – Subjects are clear and generally credible, though they may lack significant depth or nuance.
* **2 (Developing)** – Portrayal is often superficial, stereotypical, unclear, or lacks credibility.
* **1 (Deficient)** – Subjects are poorly conceived, misrepresented, confusing, or lack discernible depth.

### B. Development & Evolution *(if applicable)*

* **5 (Exceptional)** – Development (e.g., character arc, argument progression) is masterfully executed—organic, impactful, insightful.
* **4 (Accomplished)** – Clear, logical, and meaningful development is evident.
* **3 (Competent)** – Some development is present and generally logical, but may be predictable or not fully integrated.
* **2 (Developing)** – Development is minimal, forced, unconvincing, or lacks clear progression.
* **1 (Deficient)** – No meaningful development, or attempts are ineffective or counterproductive.

---

## III. Language, Style & Expressive Craft

*(Evaluates the artistry, precision, and effectiveness of language use.)*

### A. Clarity, Precision & Fitness of Language

* **5 (Exceptional)** – Language is exceptionally clear, precise, and perfectly suited to subject, audience, and purpose; demonstrates masterful command and elegance.
* **4 (Accomplished)** – Language is consistently clear, precise, and highly effective.
* **3 (Competent)** – Language is generally clear and functional; appropriate vocabulary and syntax.
* **2 (Developing)** – Language is often imprecise, vague, awkward, or inappropriate for the context.
* **1 (Deficient)** – Language is largely incomprehensible, misused, overly simplistic, or obscure.

### B. Voice, Tone & Register

* **5 (Exceptional)** – Highly distinctive, authentic, and consistent voice; tone/register perfectly aligned with content and audience.
* **4 (Accomplished)** – Clear, engaging, and consistent voice; tone/register appropriate and effective.
* **3 (Competent)** – Voice is present and mostly consistent; tone/register generally appropriate.
* **2 (Developing)** – Voice is weak or generic; tone/register often inappropriate or inconsistent.
* **1 (Deficient)** – No discernible or consistent voice; tone/register mismatched or confusing.

### C. Figurative Language, Imagery & Evocative Detail *(if applicable)*

* **5 (Exceptional)** – Uses figurative language, imagery, and sensory detail with striking originality and precision.
* **4 (Accomplished)** – Effective and often insightful use of figurative language and imagery.
* **3 (Competent)** – Adequate use, though sometimes conventional or less impactful.
* **2 (Developing)** – Limited, clichéd, forced, or ineffective use.
* **1 (Deficient)** – Lacks meaningful use, or usage is confusing/detrimental.

### D. Handling of Dialogue / Quoted Material / Interactions *(if applicable)*

* **5 (Exceptional)** – Dialogue/quotes/interactions feel authentic, purposeful, and revelatory.
* **4 (Accomplished)** – Dialogue/quotes/interactions are effective and contribute meaningfully.
* **3 (Competent)** – Functional and generally clear, but may lack nuance or strong impact.
* **2 (Developing)** – Often stilted, unnatural, or poorly integrated.
* **1 (Deficient)** – Ineffective, unrealistic, confusing, or detrimental.

---

## IV. Intellectual & Thematic Substance (or Informative Value)

*(Evaluates the depth, significance, and exploration of ideas, themes, arguments, or information.)*

### A. Depth, Significance & Nuance of Content

* **5 (Exceptional)** – Explores complex ideas with profound insight, originality, and nuance; offers significant contributions.
* **4 (Accomplished)** – Well-developed and thought-provoking content.
* **3 (Competent)** – Content is relevant and explored adequately, but may lack notable depth or originality.
* **2 (Developing)** – Content is superficial, underdeveloped, or relies on clichés/unsubstantiated claims.
* **1 (Deficient)** – Content is absent, muddled, trivial, inaccurate, or poorly conceived.

### B. Integration & Persuasiveness / Clarity of Argument / Message

* **5 (Exceptional)** – Arguments/messages are masterfully integrated, compellingly clear, logical, and subtle where appropriate.
* **4 (Accomplished)** – Well-integrated, clearly presented, and persuasive or effectively informative.
* **3 (Competent)** – Generally clear but may lack full integration, persuasive force, or complete support.
* **2 (Developing)** – Often unclear, poorly supported, inconsistently presented, or didactic.
* **1 (Deficient)** – Absent, incomprehensible, contradictory, or entirely unpersuasive/uninformative.

---

## V. Reader / Audience Impact & Engagement

*(Evaluates the work’s ability to connect with, move, inform, entertain, or otherwise engage its intended audience.)*

### A. Emotional, Intellectual, or Aesthetic Resonance

* **5 (Exceptional)** – Evokes profound and authentic responses, creating a powerful and lasting impact.
* **4 (Accomplished)** – Creates strong, genuine engagement and leaves a significant impression.
* **3 (Competent)** – Elicits appropriate responses, but impact may be somewhat superficial or fleeting.
* **2 (Developing)** – Attempts often fall flat or feel unconvincing. Limited resonance.
* **1 (Deficient)** – Fails to connect or engage; inert or elicits unintended negative reactions.

### B. Compelling Quality & Sustained Interest

* **5 (Exceptional)** – Highly engaging, sustaining intense interest through narrative, intellect, aesthetics, clarity, wit, etc.
* **4 (Accomplished)** – Consistently engaging; holds attention well.
* **3 (Competent)** – Reasonably engaging, but may have slower moments.
* **2 (Developing)** – Offers limited engagement; struggles to maintain interest.
* **1 (Deficient)** – Tedious or frustrating; fails to capture or sustain interest.

### C. Originality, Creativity & Freshness

* **5 (Exceptional)** – Striking originality and creativity in concept, execution, perspective, form, or language.
* **4 (Accomplished)** – Significant originality or clever, fresh approach; strong creative vision.
* **3 (Competent)** – Competent execution but relies on familiar conventions; shows some creativity.
* **2 (Developing)** – Largely derivative or clichéd; lacks originality.
* **1 (Deficient)** – Wholly unoriginal or an uninspired collection of overused elements.

---

## VI. Technical Execution & Adherence to Form

*(Evaluates foundational correctness, polish, and appropriate use of conventions specific to the writing form.)*

### A. Mechanics (Grammar, Syntax, Punctuation, Spelling)

* **5 (Exceptional)** – Virtually flawless; any deviations are intentional, sophisticated stylistic choices.
* **4 (Accomplished)** – Very few minor errors that do not impede readability or professionalism.
* **3 (Competent)** – Some errors present but infrequent; do not significantly hinder comprehension.
* **2 (Developing)** – Frequent errors that distract or occasionally obscure meaning.
* **1 (Deficient)** – Riddled with errors, making the text difficult to read or understand.

### B. Formatting & Form-Specific Conventions

* **5 (Exceptional)** – Formatting is impeccable, perfectly adhering to or artfully utilizing form conventions; enhances readability and professionalism.
* **4 (Accomplished)** – Formatting is clean, appropriate, and consistent; adheres well to conventions.
* **3 (Competent)** – Formatting is acceptable with minor inconsistencies that don’t seriously detract.
* **2 (Developing)** – Formatting is sloppy, inconsistent, or shows disregard for key conventions, hindering readability.
* **1 (Deficient)** – Formatting is severely problematic, absent, or inappropriate, significantly hindering comprehension.

---

# Media Types

## I. Fictional Narrative Forms

### A. Prose Fiction (Long-Form)

* **Literary Novel** — *e.g., contemporary literary fiction, classics*
* **Genre Novel**

  * *Fantasy Novel* — high fantasy, urban fantasy, grimdark, etc.
  * *Science Fiction Novel* — hard SF, space opera, cyberpunk, dystopian, etc.
  * *Horror Novel* — supernatural, psychological, body horror, slasher, etc.
  * *Thriller / Suspense Novel* — crime, legal, spy, psychological, techno-thriller
  * *Mystery / Detective Novel* — cozy, hard-boiled, police procedural
  * *Romance Novel* — contemporary, historical, paranormal, erotica sub-genres
* **Historical Fiction Novel**
* **Young Adult (YA) Novel** — across all above genres
* **Middle Grade Novel**
* **Children’s Picture Books** — text component
* **Experimental / Avant-Garde Novel**
* **Novella**

### B. Prose Fiction (Short-Form)

* **Short Story** — literary and genre
* **Flash Fiction / Micro-Fiction**
* **Vignette**

### C. Poetic Forms

* **Lyric Poetry** — sonnets, odes, elegies, free verse, etc.
* **Narrative Poetry** — epics, ballads
* **Dramatic Poetry**
* **Experimental Poetry**
* **Haiku, Tanka, and other specific forms**
* **Spoken Word Poetry** — transcripts

### D. Dramatic Forms (for Performance / Reading)

* **Stage Play Script** — full-length, one-act
* **Screenplay** — feature film, short film
* **Teleplay** — TV episode, miniseries
* **Radio Play Script**
* **Musical Libretto / Book**

### E. Graphic & Sequential Art Narratives (Textual Components)

* **Comic Book Script**
* **Graphic Novel Script**
* **Japanese Manga** — translated text or original script if available
* **Webcomic Script / Text**

### F. Interactive & Participatory Narratives

* **Video-Game Scripts / Narrative Design Docs** — dialogue, lore, quest text
* **Choose-Your-Own-Adventure / Interactive Fiction** — text-based
* **Role-Playing Game (RPG) Transcripts** — actual-play sessions

  * Standard RPG transcripts (e.g., *D\&D*, *Pathfinder* actual plays)
  * Erotic Role-Playing transcripts (ERP)
* **Role-Playing Game Scenarios / Modules** — written by GMs / designers

---

## II. Non-Fictional Forms

### A. Academic & Scholarly Writing

* Academic journal article (peer-reviewed)
* Academic monograph / book
* Dissertation / thesis
* Conference paper / proceeding
* Literature review
* Research proposal
* Textbook chapters

### B. Journalistic Writing

* News report (hard news)
* Investigative journalism piece
* Feature article (magazine or newspaper)
* Opinion piece / op-ed
* Editorial
* Column (humor, advice, political)
* Profile / interview article
* Review (book, film, music, product, restaurant, etc.)
* Blog post (informative / journalistic style)

### C. Personal & Creative Non-Fiction

* Memoir / autobiography
* Biography
* Personal essay
* Travel writing / travelogue
* Nature writing
* Food writing / culinary non-fiction
* Humor writing (non-fictional)
* Diary / journal entries (intended for an audience or of historical interest)

### D. Persuasive & Argumentative Writing

*(Non-Academic / Journalistic)*

* Political speeches / manifestos (transcripts)
* Advocacy letters / white papers
* Grant proposals
* Marketing copy / advertising text
* Business proposals / reports

### E. Informative & Technical Writing

* Technical manuals / user guides
* How-to guides / instructional content
* Scientific popularization articles / books
* Reference works (e.g., encyclopedia entries)
* FAQ documents
* Legal documents (briefs, opinions — clarity & argumentation focus)
* Medical information for laypeople
* Case studies (business, medical, social-science)

### F. Digital & User-Generated Content

*(Text-Focused)*

* Social-media posts (longer-form — Twitter threads, Substack, LinkedIn articles)
* Forum posts / online discussions (argument, clarity, contribution)
* Product descriptions (e-commerce)
* User reviews (detailed, thoughtful)
* Website content / copy (About Us, service pages)
* Email newsletters

---

## III. Specialized & Erotic Content

*(Explicitly listed for clarity; overlaps with some genres above)*

* Erotic fiction — short stories, novellas, novels across sub-genres
* Erotic poetry
* BDSM scene scripts / narratives
* Online erotic role-play logs (distinct from transcripts, often more narrative)



