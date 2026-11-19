# GSM8K 提示词核验证据

两道共同样例是 James 写信（`gsm8k-train-00004`）和 Claire 做蛋卷（`gsm8k-test-00018`）。Base 的文本块从 `Question:` 或开头指令显示，实际 token 序列在前面另有一个 BOS（`128000`）。每组这两道题的 SFT、DPO、RLVR 文本与 token ID 分别相同，以下每题共用一个后训练版本。代码块的闭合换行按文字说明解释，精确换行和 token ID 以 JSON 为准。 [四组汇总](report_zh.md) · [逐项文本、token ID、哈希和执行设置](prompt_evidence.json) · [数据许可](../NOTICE.md)

## 推理示例

原始 tokenized JSONL 的 SHA-256 与冻结清单逐阶段相符。 [从归档提取的两题输入记录](prompt_evidence.json) · [推理示例冻结清单](../cot/reports/manifests/frozen_protocol.json) · [输入准备](../cot/prepare.py) · [分词实现](../cot/tokenize_panel.py)

### cot-james

**James 写信**（`gsm8k-train-00004`）

#### Base

694 token。提示文本在冒号处结束。

```text
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
Answer:
```

#### SFT／DPO／RLVR

701 token。提示文本末尾有一个换行。

```text
<|user|>
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
<|assistant|>
```

### cot-claire

**Claire 做蛋卷**（`gsm8k-test-00018`）

#### Base

696 token。提示文本在冒号处结束。

```text
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
Answer:
```

#### SFT／DPO／RLVR

703 token。提示文本末尾有一个换行。

```text
<|user|>
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
<|assistant|>
```

## 直接答案示例

原始 tokenized JSONL 的 SHA-256 与冻结清单逐阶段相符。 [从归档提取的两题输入记录](prompt_evidence.json) · [直接答案示例冻结清单](../direct-answer/manifests/frozen_protocol.json) · [输入准备](../direct-answer/prepare.py) · [分词实现](../direct-answer/tokenize_panel.py)

### direct-answer-james

**James 写信**（`gsm8k-train-00004`）

#### Base

365 token。提示文本在冒号处结束。

```text
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:8

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
Answer:
```

#### SFT／DPO／RLVR

372 token。提示文本末尾有一个换行。

```text
<|user|>
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:8

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
<|assistant|>
```

### direct-answer-claire

**Claire 做蛋卷**（`gsm8k-test-00018`）

#### Base

367 token。提示文本在冒号处结束。

```text
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:8

Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
Answer:
```

#### SFT／DPO／RLVR

374 token。提示文本末尾有一个换行。

```text
<|user|>
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:8

Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
<|assistant|>
```

## 推理示例＋直接输出指令

在原推理示例内容前加固定指令和两个换行，按实验分词流程重建。四阶段的文本哈希、token 哈希和长度均与历史运行记录匹配。 [历史运行记录，按题号搜索](../cot-direct-answer/reports/examples.json) · [历史分词清单](../cot-direct-answer/provenance/historical_tokenization.json) · [输入准备](../cot-direct-answer/prepare.py) · [分词实现](../cot-direct-answer/tokenize_panel.py)

### cot-direct-answer-james

**James 写信**（`gsm8k-train-00004`）

#### Base

715 token。提示文本在冒号处结束。

```text
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
Answer:
```

#### SFT／DPO／RLVR

722 token。提示文本末尾有一个换行。

```text
<|user|>
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
<|assistant|>
```

### cot-direct-answer-claire

**Claire 做蛋卷**（`gsm8k-test-00018`）

#### Base

717 token。提示文本在冒号处结束。

```text
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
Answer:
```

#### SFT／DPO／RLVR

724 token。提示文本末尾有一个换行。

```text
<|user|>
Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
<|assistant|>
```

## 无示例

直接使用独立核验清单绑定的保存记录。 [原始提示词样例与 token ID](../zero-shot/reports/prompt_examples.json) · [独立核验](../zero-shot/reports/independent_audit.json) · [移除示例的实现](../zero-shot/prepare.py)

### zero-shot-james

**James 写信**（`gsm8k-train-00004`）

#### Base

31 token。提示文本在冒号处结束。

```text
Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
Answer:
```

#### SFT／DPO／RLVR

38 token。提示文本末尾有一个换行。

```text
<|user|>
Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
<|assistant|>
```

### zero-shot-claire

**Claire 做蛋卷**（`gsm8k-test-00018`）

#### Base

33 token。提示文本在冒号处结束。

```text
Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
Answer:
```

#### SFT／DPO／RLVR

40 token。提示文本末尾有一个换行。

```text
<|user|>
Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
<|assistant|>
```

## Ai2 官方 GSM8K

### ai2-rlvr-james

按 Ai2 8B RLVR 配方渲染固定版训练数据。示例和待答题在同一个用户回合，标准数值另存为 `ground_truth`。下面两个数据行经固定版 Parquet 核对。训练时某个数据行被采样的次数需要原运行日志。 [论文 `6.1](https://arxiv.org/html/2411.15124v2#S6.SS1) · [8B 复现配方](https://github.com/allenai/open-instruct/blob/f73ede5e4508b6daa6bdd1be200fb084f57a9f52/docs/tulu3.md#L306-L350) · [固定版数据集卡](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/blob/7dbd180f5440c0b90f2944e6efea934b85437a95/README.md) · [固定版 Parquet](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/resolve/7dbd180f5440c0b90f2944e6efea934b85437a95/data/train-00000-of-00001.parquet) · [数据文件核验和训练行映射](../cot/reports/manifests/data.json) · [训练模板](https://github.com/allenai/open-instruct/blob/b08997673f7c451171fe3ed39463114e7eeaf141/open_instruct/dataset_processor.py#L133-L150) · [训练预处理](https://github.com/allenai/open-instruct/blob/b08997673f7c451171fe3ed39463114e7eeaf141/open_instruct/dataset_processor.py#L434-L453)

**James 写信**（`gsm8k-train-00004`，零基行号 `6351`，`ground_truth = 624`）。 [数据浏览器第 6351 行](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/viewer/default/train?row=6351)

#### 训练生成输入

701 token。提示文本末尾有一个换行。

```text
<|user|>
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
<|assistant|>
```

**Jasper 买食材**（`gsm8k-train-00013`，零基行号 `936`，`ground_truth = 35`）。 [数据浏览器第 936 行](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/viewer/default/train?row=936)

#### 训练生成输入

741 token。提示文本末尾有一个换行。

```text
<|user|>
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer:There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer:There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer:Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer:Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer:Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer:There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer:Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer:Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?
<|assistant|>
```

### ai2-olmes-claire

Claire 做蛋卷（`gsm8k-test-00018`）按固定版本 OLMES 的 `gsm8k::tulu` 配置和 8B 原生模板重建。八条示例各占一对用户和助手回合，待答题占第九个用户回合，助手生成标记后拼接 `Answer:`。材料可复现这条输入的构造，官方当次运行的逐题请求日志未用于核验。 [论文 GSM8K 评测说明](https://arxiv.org/html/2411.15124v2#S7.SS2) · [任务配置](https://github.com/allenai/olmes/blob/e0d79fd5ad86dd46cc1a91874ecf0b3743827e39/oe_eval/configs/tasks.py#L1269-L1285) · [任务实现](https://github.com/allenai/olmes/blob/e0d79fd5ad86dd46cc1a91874ecf0b3743827e39/oe_eval/tasks/oe_eval_tasks/gsm8k.py#L50-L79) · [八条示例](https://github.com/allenai/olmes/blob/e0d79fd5ad86dd46cc1a91874ecf0b3743827e39/oe_eval/tasks/fewshot_sources.py#L1734-L1775) · [多轮拼接](https://github.com/allenai/olmes/blob/e0d79fd5ad86dd46cc1a91874ecf0b3743827e39/oe_eval/tasks/base_task.py#L424-L466) · [助手前缀拼接](https://github.com/allenai/olmes/blob/e0d79fd5ad86dd46cc1a91874ecf0b3743827e39/oe_eval/run_eval.py#L159-L175) · [模型模板](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B/blob/666943798adbde0b1aff34626007e26986a3c107/tokenizer_config.json) · [Claire 原题第 19 行](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl#L19)

#### 评测生成输入

799 token。提示文本在冒号处结束。

```text
<|user|>
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
<|assistant|>
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.<|end_of_text|>
<|user|>
Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
<|assistant|>
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.<|end_of_text|>
<|user|>
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
<|assistant|>
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.<|end_of_text|>
<|user|>
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
<|assistant|>
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.<|end_of_text|>
<|user|>
Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
<|assistant|>
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.<|end_of_text|>
<|user|>
Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
<|assistant|>
Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.<|end_of_text|>
<|user|>
Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
<|assistant|>
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.<|end_of_text|>
<|user|>
Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
<|assistant|>
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.<|end_of_text|>
<|user|>
Question: Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?
<|assistant|>
Answer:
```

## 运行设置记录

JSON 中的 `runtime` 保留四组资产清单和配置文件的 SHA-256、实际记录的硬件和核心软件版本、模型修订、数据修订与推理设置。推理示例和直接答案示例的源资产清单可以直接查看。推理示例＋指令和无示例的源清单哈希分别列在这两组的冻结清单和独立核验清单。 [四组摘录](prompt_evidence.json) · [推理示例资产清单](../cot/reports/manifests/assets.json) · [直接答案示例资产清单](../direct-answer/manifests/assets.json) · [推理示例＋指令冻结清单](../cot-direct-answer/provenance/historical_freeze.json) · [无示例独立核验](../zero-shot/reports/independent_audit.json)
