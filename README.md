
<header>
    <h1>Text Classification Assignment</h1>
    <p>This repository contains the implementation of an assignment for a Natural Language Processing (NLP) course at the University of Haifa. The assignment focuses on text classification using Israeli parliamentary protocols.</p>
</header>

<section>
    <h2>Implementation Overview</h2>
    <p>The implementation follows the assignment guidelines provided by the course. Here's a brief overview of the tasks:</p>
    <ol>
        <li><strong>Define Classes</strong>: Utilize the Knesset corpus to define classes based on the source of the protocol.</li>
        <li><strong>Partitioning</strong>: Divide the data into classification units consisting of 5 sentences each.</li>
        <li><strong>Class Balancing</strong>: Ensure an equal distribution of data between the classes.</li>
        <li><strong>Feature Vector Creation</strong>: Create feature vectors using Bag of Words or TF-IDF for text classification.</li>
        <li><strong>Training</strong>: Train classifiers such as KNearestNeighbors and SVM on the feature vectors.</li>
        <li><strong>Classification</strong>: Classify new text chunks into the appropriate classes.</li>
    </ol>
</section>

<section>
    <h2>How to Run</h2>
    <h3>Prerequisites</h3>
    <ul>
        <li>Python installed on your system.</li>
        <li>Required Python packages installed.</code>.</li>
    </ul>
    <h3>Steps to Run</h3>
    <ol>
        <li><strong>Define Classes</strong>: Refer to the provided Knesset corpus and assign classes based on the source of the protocol.</li>
        <li><strong>Partitioning</strong>: Use the script <code>partitioning_script.py</code> to partition the data into classification units.</li>
        <li><strong>Class Balancing</strong>: Implement down-sampling to balance the classes.</li>
        <li><strong>Feature Vector Creation</strong>: Utilize Bag of Words or TF-IDF to create feature vectors.</li>
        <li><strong>Training</strong>: Train classifiers using KNearestNeighbors and SVM on the feature vectors.</li>
        <li><strong>Classification</strong>: Classify new text chunks using the trained classifiers.</li>
    </ol>
    <h3>Example</h3>
    <pre><code>python knesset_protocol_classification.py</code></pre>
</section>

<footer>
    <p>For any issues or inquiries, please contact Faisal Omari - <a href="mailto:faisalomari321@gmail.com">faisalomari321@gmail.com</a></p>
</footer>
