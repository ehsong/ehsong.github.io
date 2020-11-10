# Mood Metric: Detecting Mood at the Workplace Using Lexicon-Based Approach

Detecting mood at the workplace has become a central task for employers today to increase productivity and reduce economic costs at the workplace. It has been found that negative emotions, such as anxiety, stress, and depression, lead to impaired productivity and decreased job retention.
Technology has led to several solutions that enable detecting mood at the workplace. The company I consulted during my time as a Data Science fellow at [Insight](https://insightfellows.com/) was a startup that was creating an add-on to slack, a workplace chat platform. The add-on automatically assesses mood and engagement through using natural language processing on employee conversation data and give suggestions to managers using a chatbot. The goal of the company was to offer a convenient tool for employees to track mood at the workplace.
Because the company was open to any type of emotion detection, I was able to frame my project in a free manner. Starting from the fact that the company was only offering only negative and positive mood metrics, I decided to build a tool that captures a wider range of emotions from text data. This is how I came about building the Mood Metric that captures eight ranges of emotions.

## Detecting Emotions: A Lexicon-Based Approach

To build the metric, I chose an unsupervised learning approach called a lexicon-based approach, which is a method that relies on the ratio of lexicon word occurrences in a document over the total count of words in the document.
However, with several lexicons that exist, I had to decide on which lexicon to use. One of the well-known lexicons is the [National Research Council (NRC)](https://medium.com/r?url=https%3A%2F%2Fsaifmohammad.com%2FWebPages%2FNRC-Emotion-Lexicon.htm) lexicon that has been used to detect emotions in books. This lexicon contains about 14K words, and each word was labeled one out of seven emotions - anger, anticipation, disgust, fear, joy, sadness, surprise, and trust by crowdsourcing annotations using Amazon Mechanical Turk.
Another lexicon I considered was [DepecheMood](https://medium.com/r?url=https%3A%2F%2Fgithub.com%2Fmarcoguerini%2FDepecheMood), which is a lexicon that contains 37K terms over 8 ranges of emotions - afraid, amused, angry, annoyed, don't care, happy, inspired, and sad. This lexicon also included scores for each emotion category, which was derived from crowdsourcing emotion labels for 25.3K news articles and obtaining the word-emotion matrix.
To compare the performance of the two lexicons, I tested the performance of the two lexicons over the [International Survey On Emotion Antecedents And Reactions (ISEAR)](https://medium.com/r?url=https%3A%2F%2Fwww.unige.ch%2Fcisa%2Fresearch%2Fmaterials-and-online-research%2Fresearch-material%2F) data set which includes 7K self-written sentences across 7 range of emotions - joy, fear, anger, sadness, disgust, shame, and guilt.

![](/images/medium_fig1.png#center)
*Averaged F1-scores of NRC and DM on Categories "Anger", "Sadness", and "Joy"*

To compare the overall performance, I averaged the f1-scores obtained for three emotion categories - anger, sadness, and joy. I found that although the performance of the two lexicons is similar, DepecheMood had a slightly better performance. After comparing the overall performance, I concluded that it is plausible to use the DepecheMood lexicon for this project.

## To Include or Not to Include Emoji Translations

After examining performance, I settled on using the DepecheMood lexicon for the task. However, before obtaining the emotion score for the text corpus, I had to deal with the question of whether or not to include emojis in the corpus. For example, the emoji 😞 was translated to "disappointed" in the corpus. This question is important because the lexicon approach gives equal weights on the lexicon term that appears in the corpus.
To answer the question, I examined emoji usage over the corpus and also examined emoji usages in sentences where text sentiment and emoji sentiment did not match. The emoji usage was low at 0.01%, reflecting that including them in the corpus would have a negligible effect on classifying emotions. Also, I found that people do use emojis to convey emotions, which implies that emojis should have equal weight as textual expressions.
More importantly, the mood metric was better at accurately capturing emotions when emojis were included. For example, messages like "trouble in paradise!!😞😞" convey disappointment, but when emojis are left out, the lexicon would capture only "paradise" and "trouble" and assign a higher score for the "inspired" category. However, when including the disappointed emojis, the score decreased.

![](/images/medium_fig2.png#center)
*Scores for Inspired With or Without Including Emoji*

## Building the Mood Metric Using DepecheMood Lexicon

Now, finally building the mood metric! This required several steps: 1) Text preprocessing, which included removing stop words, punctuations, special characters 2) tokenizing and 3) looping each term in the DepecheMood lexicon through the tokenized message and obtaining emotion scores for eight categories for each token when it is matched with the term in the lexicon, 4) averaging the score in each emotion category and normalizing the scores 5) finding the max score and assigning the message to an emotion class and 6) finding the proportion of messages that were assigned to each class out of total number of messages. For the code snippets, please see my github repo [here](https://github.com/ehsong/mood-metric).
When applied to the message data I received from the company, the majority of the messages were positive, being in the "inspired" category. The second highest was "amused," and the third was "annoyed."

![](/images/medium_fig3.png#center)
*117 Users, 4087 Messages*

I also checked whether some temporal dimensions can be captured through the metric, and found that it does! For instance, the amount of angry and annoyed moods peak during the mid-week but subdues towards the end of the week.

![](/images/medium_fig4.png#center)
*% of Angry and Annoyed Messages, 117 Users, 4087 Messages*

## Tracking Mental Health at the Workplace Through Slack
The company was happy with the results I delivered but also expressed interest in detecting a range of negative emotions among employees, such as anxiety and depression. The motivation behind their task was to go beyond assessing mood and evaluate mental health, which could be a valuable piece of information for managers.
For this task, I used a [depression lexicon](https://github.com/halolimat/Social-media-Depression-Detector), which has been constructed by a group of health informatic researchers at Cornell and Wright State University. This lexicon included 1,620 terms that are likely to be expressed by patients with depressive symptoms.
To detect depressive mood in text, I combined the lexicons with topic modeling, which is conventionally known as guided LDA or semi-supervised topic modeling. Compared to unsupervised topic modeling which begins with the assumption that there is a certain distribution of words across each N topics and each document can be described as a distribution of topics, the guided LDA assumes that certain words are representative of a topic in the corpus and guides the model to assign topic classes to each document in the corpus.
Assuming that people are less likely to discuss depressive symptoms on the slack channel, I subsetted the data to messages that had negative sentiment (Vader scores), and the sentence that used the pronoun "I." I used the second criterion because it has been found that people with depressive symptoms are more likely to use the first pronoun than the second and the third (De Choudhury et al., 2013). This resulted in about 10% of the messages out of 37K. I also excluded terms related to suicidal thoughts, assuming that people are less likely to discuss very private feelings on work channels.
However, the results were not very informative. I suspected if it was because the lexicon terms rarely occur in conversations, and I found that this was exactly the case! Out of 1,620 terms and expressions in the lexicon, there were only 8 terms that matched in 3.7K messages. Also, for the matched cases, the context under which the term was used was not related to depressive mood. For instance, the word "failure," which occurred more than 10 times referred to failures in the computing system, rather than one's goals or aspirations.
Overall, my conclusion was that uncovering mental health through slack conversation data may be infeasible. It may be easier on social media platforms, where people discuss their feelings more freely. For instance, researchers used the lexicon and guided topic modeling to detect depressive symptoms from tweets of users who self-claimed to be depressed (Yazdavar, 2011). I advised the company that assessing mental health through the content of the text itself may not be promising. Given the finding that people with depressive symptoms are likely to withdraw from social engagement, perhaps behavioral patterns, such as the gradual decline in message volumes at the user-level, although indirect, can be an alternative way to detect depressive mood.

## Cited Works
De Choudhury, M., Gamon, M., Counts, S., & Horvitz, E. (2013). Predicting depression via social media. In Seventh international AAAI conference on weblogs and social media.[Link](aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/viewFile/6124/6351)  
Mohammad, S. (2011). From Once Upon a Time to Happily Ever After: Tracking Emotions in Novels and Fairy Tales. Proceedings of the 5th ACL-HLT Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities, pages 105–114.[Link](https://www.aclweb.org/anthology/W11-1514/)  
Staiano, J., & Guerini, M. (2014). Depechemood: a lexicon for emotion analysis from crowd-annotated news. arXiv preprint arXiv:1405.1605.[Link](https://arxiv.org/pdf/1405.1605.pdf)  
Yazdavar, A. H., Al-Olimat, H. S., Ebrahimi, M., Bajaj, G., Banerjee, T., Thirunarayan, K., & Sheth, A. (2017). Semi-supervised approach to monitoring clinical depressive symptoms in social media. In Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017 (pp. 1191–1198).[Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5914530/)  
