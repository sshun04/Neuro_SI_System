
脳神経科学分野において、特定の脳機能をその時点で明らかになっている脳神経科学的事実をもとにモデル化を行い実装してシミュレーションを行うことで、その結果の課題から、脳活動における明らかになっていない機能を推定する
計算論的神経科学の分野においては、脳の神経活動をモデル化し、その活動を応用して脳機能の特性を生かしたシステムを実装したり、あるいはそこから神経のさらなる機能面の課題や、未知の機能の推定を行う学問である。[計算論的神経科学の良い感じの論文]

この分野において様々な論文が出ているものの、多くのコンピュータショナルニューロサイエンスの領域においては、それらは多くの場合、計算機科学的な方の領域に寄り過ぎており、また一方、脳神経科学の領域においては、それらは生理学的なものに注意を向けており、両者、計算機科学的な面と神経科学的な面を行き来するような形での研究はあまり多くありません。

特に言語学、リングイスティックスというまた別の領域が関わってくる人の言語処理基盤の領域においては、多くの脳活動が言語学、人文学的な、例えば言語処理のタスクなどの言語学的な特性と、脳神経科学的な生理学的な機能面といった場所に注意が向けられており、ここにコンピュテーショナルニューロサイエンスを適用した形で、言語学、コンピュテーショナルニューロサイエンス、そして神経科学、これらを統合した形で行われている研究というのはほぼ存在しない。

今回の研究では、人の同時通訳、人間が行える言語処理活動の中で最も認知負荷が高く、また非常に困難な課題である同時通訳における人の脳機能のシミュレーションと再現、そのためのシステムアーキテクチャの検証といった観点からシミュレーションを行い、それを通じて人の同時通訳の脳機能に関わる機能やそこにあるメカニズムを解明することを目的とする。

同時通訳の前提として、世の中には通訳の領域においては、2つのものがある。逐次通訳と同時通訳である。

逐次通訳とは、人が話した文章を都度言い終わるのを待ち、その後に通訳をする。文章が完了したことを待った上で通訳を行い、また相手が話を終えるのを待って、その後通訳を行う。その通訳者が通訳を行っている間、話し手側は特に話すことを止めず、相互に通訳を行うものである。元の話者として通訳者が交互に話す形で通訳を行うものが逐次通訳である。

同時通訳とは、逐次通訳に対して、話者が話し終わるのを待たずに、同時通訳を開始して話すものである。このタスクにおいては、通訳者は元の話者が話している途中、適切なタイミングで、通訳が開始可能な最小の単位から通訳を開始し、可能な限り遮らず、そしてスムーズに通訳を進行することが求められる。文章が完了するのを待って、その後に通訳を行うという、一つ一つのタスクを独立して行える逐次通訳に対し、同時通訳は耳で元の言語の声を聞きながら、それを一方のもう一方の言語へと翻訳するというタスクを同時に行う必要があり、非常に高度なスキルが求められる。

通訳の現場においては、この同時通訳への需要は高く、特にカンファレンスや重要な国際会議など、様々な場面で同時通訳者が必要とされている。しかし一方で、この同時通訳というのは非常に高度な認知負荷がかかるタスクであり、これを実際に高いパフォーマンスで行えるものはなかなか限られている。ソースは多々あるが、一般的に同時通訳においての精度の世界においては7割程度の精度を目指すということがセオリーとして謳われている部分もある。

これらでは完璧な同時通訳というものを行うことはなかなか難しいのである。実際の同時通訳の現場では、15分から30分で人を交代して通訳を行う形が一般的である。

この同時通訳というタスクにおいては、できる人間は高度な言語処理能力、そしてトレーニングを必要とする。また、この同時通訳者は、一般の逐次通訳者や一般のバイリンガルなどと比べても、異なる脳活動の能力、脳機能を持っており、特定の機能が非常に強化されているという傾向があることが過去の調査から明らかになっている。[引用:同時通訳時の脳活動を研究した論文をいろいろ]

今回の研究では、過去の論文、過去の同時通訳者の認知タスクや神経タスクなどの課題から明らかになっている脳機能の脳の不活化、またそこに関わる神経ネットワークやそこに存在する各種の機能のモデリングを行い、脳機能をベースにそれらの各脳機能のモデリングを行い、それらに対して言語処理タスクを行った際の脳活動、並びにそのパフォーマンスという部分をシミュレーションしていることを目的としている。

この研究においては、当然実際の脳活動とは大きく異なる部分が、計算機でモデル化したものでは実際の活動とは異なる部分があるため、多くの仮定を用いる。これらの仮定は、過去のコンピュテーショナルサイエンスのモデルの実装などにおいて一般的に用いられている仮定を基に、実装していく方針である。[引用 : ]