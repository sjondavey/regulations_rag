from regulations_rag.embeddings import get_ada_embedding, \
                                           get_closest_nodes
import pandas as pd                        

class TestEmbeddings:
    question = "Who can trade gold?"
    question_embedding = [0.0027741477824747562, -0.022562408819794655, -0.0015197647735476494, -0.02782389335334301, -0.04362153634428978, 0.01480864081531763, -0.05221925303339958, -0.008901008404791355, -0.002091736998409033, -0.02829861454665661, 0.025397544726729393, 0.01588994823396206, -0.011248237453401089, -0.020795393735170364, 0.0066625699400901794, 0.014492160640656948, 0.01939760521054268, 0.004453801084309816, 0.010694397613406181, -0.012461411766707897, -0.024131624028086662, 0.026478853076696396, 0.0060230158269405365, 0.008841668255627155, -0.009210895746946335, -0.0004137320793233812, 0.015336108393967152, -0.003540623467415571, 0.0013079537311568856, -0.022733835503458977, 0.02307668887078762, 0.00083446956705302, -0.036738090217113495, -0.005610932130366564, -0.026333799585700035, -0.004137320909649134, 0.00807684101164341, 0.010931757278740406, 0.024645904079079628, 0.00488566467538476, 0.015006441622972488, 0.0040614972822368145, 0.0025631608441472054, 0.0017324000364169478, -0.010054843500256538, -0.0022598672658205032, -0.0010607035364955664, 0.005568075459450483, -0.021256927400827408, 0.012118558399379253, 0.005347198806703091, 0.013377886265516281, -0.006988940294831991, -0.0030955730471760035, -0.016720710322260857, -0.0025664574932307005, -0.009336168877780437, 0.02045254036784172, 0.004437317606061697, -0.0017620701109990478, 0.019173432141542435, 0.01544160209596157, -0.011248237453401089, 0.025015130639076233, 0.010430663824081421, -0.02405250445008278, 0.02054484747350216, 0.014531721360981464, -0.005162585061043501, 0.008287828415632248, 0.016733895987272263, 0.015692148357629776, -0.01505918800830841, 0.0005546647007577121, 0.006962566636502743, -0.02808762714266777, -0.009955943562090397, -0.00885485578328371, -0.012692179530858994, -0.016747083514928818, 0.019911887124180794, -0.01218449231237173, -0.016786642372608185, 0.002441184129565954, 0.01321964617818594, -0.00025569795980118215, 0.01644378900527954, -0.006599932909011841, -0.010733957402408123, -0.030250243842601776, 0.01565258949995041, 0.006230705883353949, 0.031964510679244995, 0.01930529810488224, -0.007114213425666094, 0.01165702473372221, 0.0026109626051038504, 0.013885573484003544, 0.003626336809247732, -0.024210743606090546, 0.00324227474629879, -0.0071471803821623325, -0.02310306206345558, -0.004872478079050779, -0.01519105490297079, -0.008828481659293175, -0.0044801742769777775, 0.004582371097058058, 0.010694397613406181, -0.0012156469747424126, -0.007707614451646805, 0.011762518435716629, 0.00663949316367507, -0.062030140310525894, 0.0033428233582526445, -0.016984444111585617, -0.02098000794649124, -0.026426106691360474, 0.0003626336983870715, -0.010911976918578148, 0.015507535077631474, -0.03233373910188675, 0.003566996892914176, 0.0041175405494868755, 0.011544938199222088, -0.010008689947426319, -0.03816225007176399, -0.019991006702184677, -0.0170635636895895, 0.006342792883515358, 0.0030626063235104084, 0.016140496358275414, 0.006095542572438717, -0.0021955822594463825, -0.027428293600678444, 0.04135342687368393, -0.01989869959652424, -0.003738423576578498, -0.04312044382095337, -0.005779062397778034, -0.0003084446652792394, 0.030461229383945465, 0.02133604697883129, 0.0013986121630296111, -0.010635057464241982, 0.018856951966881752, -0.0077471742406487465, 0.015336108393967152, -0.013911946676671505, -0.02116462029516697, -0.006708723027259111, -0.004476877860724926, 0.03106781654059887, -0.013516346924006939, 0.0012296579079702497, 0.039718277752399445, -0.005703238770365715, 0.017525097355246544, -0.014558094553649426, -0.02171846106648445, -0.028641467913985252, -0.01889651268720627, 0.01668114960193634, 0.009382322430610657, -0.01806575059890747, 0.028931574895977974, 0.014360293745994568, 0.020505286753177643, -0.003590073436498642, 0.0036988635547459126, 0.00801750086247921, 0.00348458020016551, -0.010311983525753021, 0.010074622929096222, 0.007569154258817434, 0.024250304326415062, 0.01906793937087059, -0.023393169045448303, 0.004206550773233175, -0.008439474739134312, -0.00043062749318778515, 0.005475768819451332, 0.007153773680329323, 0.017854765057563782, -0.01190757192671299, 0.008564748801290989, 0.0016928400145843625, -0.021731648594141006, 0.014966880902647972, -0.02316899597644806, -0.0100218765437603, 0.023234929889440536, -0.015520721673965454, 0.006497736554592848, -0.6553252339363098, -0.021797580644488335, -0.013635026291012764, -0.0013887222157791257, 0.01983276568353176, 0.01871189847588539, 0.005848292261362076, 0.018369045108556747, -0.02587226592004299, 0.01854047179222107, -0.009290015324950218, 0.02832498773932457, -0.024751396849751472, -0.011373511515557766, 0.01171636488288641, -0.03201725706458092, -0.0006440868601202965, -0.015718521550297737, -0.005877962335944176, 0.003801060374826193, 0.013061406090855598, 0.019318485632538795, 0.00825486145913601, 0.007859260775148869, 0.027665654197335243, -0.005027421750128269, -0.0100218765437603, -0.03536667302250862, 0.012566905468702316, 0.0004640062979888171, -0.016747083514928818, 0.02961728163063526, -0.0016582249663770199, 0.005386758595705032, 0.03861059620976448, -0.0024214040022343397, -0.01528336200863123, 0.015573468990623951, -0.010681210085749626, 0.015599842183291912, -0.02166571468114853, -0.004246111027896404, 0.024870077148079872, -0.005188958253711462, 0.007668054196983576, 0.004697754513472319, -0.0028433778788894415, 0.025357984006404877, 0.007846074178814888, -0.02033386006951332, -0.008360355161130428, 0.0015304790576919913, -0.011235050857067108, -0.012632839381694794, 0.0004096112388651818, 0.01047681737691164, 0.013035032898187637, -0.005647195503115654, -0.009501002728939056, 0.01792069710791111, 0.003530733520165086, 0.020136060193181038, -0.0056900521740317345, -0.00661641638725996, 0.00677795335650444, -0.04053585231304169, -0.026544785127043724, 0.034971073269844055, 0.002286240691319108, -0.009672429412603378, -0.009257049299776554, 0.005568075459450483, -0.007635087706148624, 0.02548985183238983, 0.018026191741228104, 0.014017440378665924, 0.014887761324644089, -0.0046318210661411285, -0.0037944670766592026, 0.020162433385849, -0.01995144598186016, -0.008479034528136253, 0.007687834091484547, -0.010753736831247807, 0.0443863645195961, -0.009019688703119755, -0.013700960204005241, -0.029248055070638657, 0.02670302614569664, 0.0073186070658266544, -0.01747235096991062, 0.026478853076696396, 0.012494378723204136, -0.0018148167291656137, 0.00683729350566864, -0.004770281258970499, -0.009606496430933475, -0.016549283638596535, 0.0073845405131578445, -0.003890070365741849, 0.01679982990026474, 0.004734017886221409, -0.0028400812298059464, 0.016114123165607452, 0.012751518748700619, 0.01744597777724266, -0.032966699451208115, 0.017577843740582466, 0.055014826357364655, -0.04459735006093979, 0.018843764439225197, 0.0009172984282486141, -0.012520751915872097, 0.0006568614626303315, -0.016997629776597023, -0.02219318225979805, 0.012118558399379253, -0.026755772531032562, -0.011169117875397205, -0.012599872425198555, 0.011624057777225971, 0.012164711952209473, 0.0312524288892746, -0.004209847655147314, 0.009738363325595856, 0.011762518435716629, -0.024514038115739822, -0.025265678763389587, 0.002202175557613373, -0.002906014444306493, -0.005027421750128269, -0.011057030409574509, 0.007984534837305546, -0.009929569438099861, 0.02879970893263817, 0.004170287400484085, 0.010410883463919163, 0.009665836580097675, 0.013450413011014462, -0.012065812014043331, -0.009850449860095978, -0.009111995808780193, -0.007773547433316708, -0.025925012305378914, -0.00010075447062263265, -0.010041656903922558, 0.0023637122940272093, 0.0061746626161038876, -0.024395357817411423, -0.016615215688943863, 0.010938351042568684, -0.012125152163207531, -0.02417118288576603, 0.01871189847588539, -0.008571341633796692, -0.018487725406885147, -0.011070217937231064, -0.029432669281959534, 0.004612041171640158, -0.010371323674917221, -0.0007186740403994918, -0.005535108968615532, -0.014769081026315689, -0.012837233021855354, 0.012065812014043331, -0.00255986419506371, -0.019358046352863312, 0.003952707163989544, 0.01393832080066204, -0.021771207451820374, 0.007832887582480907, -0.01530973520129919, -0.009922976605594158, 0.02328767627477646, -0.007490034215152264, -0.014676774851977825, 0.01954265870153904, -0.007964754477143288, -0.0173009242862463, 0.006428506225347519, 0.004134024027734995, 0.002228548750281334, -0.025714024901390076, -0.0099625363945961, 0.0266107190400362, -0.004486767575144768, -0.02402612939476967, 0.02381514385342598, -0.0016104232054203749, 0.00872298888862133, -0.006599932909011841, 0.0431995615363121, 0.020623967051506042, -6.110429239924997e-06, 0.012118558399379253, -0.02018880657851696, -0.02505469135940075, 0.01641741581261158, 0.0042032538913190365, 0.051507171243429184, 0.010325170122087002, -0.017525097355246544, -0.003237329889088869, -0.01529654860496521, 0.020254740491509438, -0.016865763813257217, 0.010562530718743801, -0.022826142609119415, 0.014373480342328548, 0.019239366054534912, 0.007931787520647049, -0.02384151704609394, 0.01918661966919899, 0.02319536916911602, 0.018026191741228104, 0.031727150082588196, 0.016747083514928818, 0.01600862853229046, 0.02231186255812645, -0.009105402044951916, -0.0059274123050272465, -0.030698589980602264, 0.01004824973642826, -0.021758021786808968, -0.005060388240963221, 0.020584406331181526, 0.00807684101164341, -0.02980189584195614, -0.00527467206120491, -0.009210895746946335, -0.02334042266011238, -0.037687528878450394, -0.009026282466948032, 0.003913147374987602, 0.00505379494279623, -0.012270205654203892, 0.0072328937239944935, -0.004368087742477655, 0.026531599462032318, -0.01289657223969698, 0.000667987740598619, -0.01478226762264967, -0.00965264905244112, 0.004555997904390097, 0.03526118025183678, -0.010536156594753265, -0.0010458684992045164, 0.019384419545531273, -0.01751190982758999, 0.023973383009433746, -0.0031400781590491533, 0.0026406326796859503, -0.025265678763389587, 0.006125212647020817, 0.004984565079212189, -0.004120837431401014, 0.025634905323386192, -0.005904335994273424, -0.01170317828655243, 0.04306769371032715, 0.012382292188704014, -0.002960409503430128, -0.0196877121925354, -0.005258188582956791, 0.03407438099384308, 0.015903135761618614, 0.023656902834773064, 0.010087809525430202, -0.007213113829493523, -0.0047274245880544186, -0.01542841549962759, 0.00878232903778553, 0.004190067294985056, 0.0015824015717953444, 0.020531659945845604, -0.011109777726233006, -0.006052685901522636, -0.01049000397324562, 0.00024848649627529085, 0.0015560281462967396, -0.021362420171499252, -0.038004010915756226, 0.017459163442254066, 0.029485415667295456, -0.02183714136481285, -0.0242239311337471, -0.0023637122940272093, 0.007490034215152264, -0.006372462958097458, 0.05121706426143646, 0.0017192133236676455, -0.009309795685112476, -0.02961728163063526, -0.012678992003202438, -0.00327853811904788, -0.014571281149983406, 0.020505286753177643, -0.010865824297070503, -0.029168935492634773, -0.014175680465996265, -0.0022400871384888887, -0.029986510053277016, 0.01679982990026474, -0.02396019734442234, 0.025001944974064827, 0.006936193443834782, -0.016021816059947014, -0.012164711952209473, -0.01466358732432127, -0.020874513313174248, 0.009481222368776798, -0.018355857580900192, -0.020149245858192444, -0.009501002728939056, -0.0010170226451009512, -0.009850449860095978, -0.009665836580097675, 0.019964633509516716, 0.01629873551428318, -0.0016656424850225449, -0.003952707163989544, -0.03085683099925518, -0.017788831144571304, 0.007668054196983576, 0.10306708514690399, 0.0026076659560203552, -0.024382170289754868, 0.013259205967187881, -0.009751549921929836, -0.00873617548495531, 0.009870229288935661, -0.018698710948228836, 0.004773578140884638, 0.006240596063435078, 0.007780140731483698, -0.003501063445582986, -0.014821827411651611, 0.03863697126507759, 0.00012393418001011014, 0.0136614004150033, -0.009883416816592216, -0.019608592614531517, 0.015547094866633415, -0.012197678908705711, 0.010397696867585182, 0.018698710948228836, 0.0018444868037477136, 0.02885245531797409, 0.0007598823867738247, 0.006003235932439566, 0.03228099271655083, 0.00248733744956553, -0.012468005530536175, -0.0053274184465408325, -0.0013236129889264703, -0.00344831682741642, 0.0021692088339477777, 0.015467975288629532, -0.0020670120138674974, -0.020505286753177643, -0.010958130471408367, 0.002490634098649025, 0.007206520531326532, -0.019199805334210396, 0.030276617035269737, -0.0009535617427900434, 0.012164711952209473, -0.02018880657851696, 0.010588903911411762, -0.0016738841077312827, 8.380752842640504e-05, 0.01768333651125431, -0.004786764737218618, 0.007186740171164274, 0.00994934979826212, -0.007437287364155054, -0.022984381765127182, -0.003138429718092084, 0.0029142561834305525, 0.004361494444310665, 0.0054625822231173515, -0.015586655586957932, -0.028773335739970207, -0.026689838618040085, -0.019555846229195595, -0.0005538405384868383, -0.0011497136438265443, 0.002760960953310132, -0.012468005530536175, -0.011024064384400845, -0.01877783238887787, -0.002978541189804673, -0.0026785442605614662, -0.008274641819298267, 0.0007833711570128798, -0.03183264285326004, -0.005920819006860256, -0.00010379359446233138, 0.017881138250231743, 0.007206520531326532, 0.024448104202747345, -0.006085652858018875, -0.008452661335468292, 0.019107498228549957, 0.010648244060575962, -0.02360415644943714, -0.007246080320328474, -0.027692027390003204, -0.0021181104239076376, 0.004895554855465889, -3.6881494452245533e-05, -0.004321934189647436, -0.01505918800830841, 0.029670029878616333, -0.002764257602393627, 0.012830639258027077, 0.008597714826464653, -0.012217458337545395, 0.0017373450100421906, 0.010885603725910187, 0.016232803463935852, 0.035524915903806686, -0.006349386181682348, -0.0009906493360176682, -0.02782389335334301, -0.0053669787012040615, -0.017010817304253578, -0.037265557795763016, -0.01480864081531763, 0.0065768565982580185, 0.008657054975628853, 0.003979080356657505, -0.013727333396673203, -0.0027692026924341917, -0.003599963616579771, -0.03283483162522316, -0.009296609088778496, -0.021652527153491974, -0.009441662579774857, 0.02490963786840439, 0.016430603340268135, 0.05047861114144325, 0.010694397613406181, 0.0073186070658266544, -0.005449395161122084, -0.005821919068694115, 0.00977132935076952, 0.004503251053392887, -0.00819552130997181, -0.011432851664721966, 0.008868042379617691, -0.03307219222187996, 0.030171122401952744, 0.0021922853775322437, 0.013272392563521862, 0.009487816132605076, -8.504377910867333e-05, -0.021744834259152412, -0.025120625272393227, -0.009388916194438934, -0.014729521237313747, 0.001890640240162611, -0.019489912316203117, -0.009632869623601437, -0.008617495186626911, -0.010503190569579601, -0.009125182405114174, -0.035973262041807175, -0.012883385643362999, -0.025529412552714348, -0.001902178511954844, 0.005917522590607405, -0.0030609581153839827, 0.023643717169761658, -0.017551470547914505, 0.013489972800016403, -0.010127370245754719, 0.003507656743749976, 0.019358046352863312, -0.014940507709980011, 0.009936163201928139, 0.00031606823904439807, 0.036738090217113495, 0.029248055070638657, 0.03932267799973488, 0.013450413011014462, 0.01939760521054268, 0.010681210085749626, 0.01155153103172779, -0.01490094792097807, -0.005159288179129362, 0.0023290973622351885, -0.01133395079523325, 0.014439414255321026, -5.36223960807547e-05, 0.006135102827101946, 0.02452722378075123, 0.005498845595866442, -0.02360415644943714, 0.012204271741211414, 0.004483471158891916, -0.014716334640979767, -0.0393754243850708, -0.027270052582025528, 0.002179098781198263, -0.03344142064452171, -0.008050467818975449, -0.00879551563411951, -0.03352053835988045, -0.022377794608473778, 0.055806029587984085, 0.008597714826464653, 0.0024065689649432898, 0.004648304544389248, 0.03301944583654404, -0.02328767627477646, 0.02098000794649124, -0.0027626093942672014, 0.01591632142663002, 0.0003061782044824213, 0.004183473996818066, -0.01739322952926159, -0.009441662579774857, 0.018237177282571793, 0.005907632410526276, 0.001931848586536944, -0.014109747484326363, 0.016338296234607697, -0.009243862703442574, 0.023947009816765785, -0.017762457951903343, 0.02260196954011917, 0.005680162459611893, -0.009955943562090397, -0.021626153960824013, -0.024777770042419434, -0.01703719049692154, -0.008320794440805912, 0.02469865046441555, -0.004984565079212189, 0.013094373047351837, 0.02579314447939396, -0.019700899720191956, -0.025015130639076233, 0.007727394346147776, 0.010265829972922802, 0.027164559811353683, 0.008749362081289291, 0.030988696962594986, -0.009276828728616238, 0.00983066949993372, 0.01047681737691164, 0.0012774595525115728, -0.002327448921278119, -0.010377916507422924, 0.009151555597782135, 0.0013186680153012276, -0.02436898462474346, 0.0009774626232683659, -0.0057955458760261536, 0.008690021932125092, -0.008037281222641468, -0.00341864675283432, 0.021678900346159935, -0.008901008404791355, -0.001061527756974101, 0.00017163286975119263, -0.025120625272393227, -0.025542598217725754, 0.03328317776322365, -0.004677974618971348, 0.006128509528934956, 0.01217130571603775, -0.014966880902647972, -0.02903706766664982, 0.025740398094058037, -0.016404230147600174, 0.03333592787384987, 0.013298766687512398, -0.009613089263439178, 0.006088949274271727, -0.00520873861387372, -0.023234929889440536, 0.01951628550887108, 0.0010598793160170317, 0.027164559811353683, 0.014083373360335827, -0.004404351115226746, 0.008432881906628609, 0.00696916040033102, -0.04496657848358154, -0.009705396369099617, -0.024329423904418945, 0.021006381139159203, -0.018738271668553352, 0.0037911704275757074, 0.023182183504104614, 0.018355857580900192, 0.007002126891165972, 0.02071627415716648, -0.007635087706148624, 0.0023027239367365837, 0.00011847407586174086, -0.009019688703119755, 0.02009649947285652, 0.007588934153318405, -0.01126142404973507, -0.01603500172495842, -0.010753736831247807, -0.0023159105330705643, -0.005419725086539984, 0.0019104202510789037, 0.04691820591688156, -0.01859321817755699, -0.007140587083995342, -0.011722957715392113, 0.0017554766964167356, 0.009125182405114174, 0.011432851664721966, 0.0017010816372931004, 0.034997448325157166, 0.014228426851332188, -0.011518565006554127, 0.0025071173440665007, 0.012665805406868458, 0.012632839381694794, -0.0077471742406487465, 0.011287798173725605, 0.02596457302570343, -0.03705456852912903, 0.02806125394999981, -0.017525097355246544, -0.018237177282571793, -0.004308747593313456, 0.008129588328301907, -0.01677345670759678, -0.0014365238603204489, -0.0021527253556996584, -0.0023884372785687447, 0.031199684366583824, 0.005343901924788952, 0.005752689205110073, 0.0002800108923111111, 0.0008225191268138587, -0.012415259145200253, 0.012698772363364697, 0.013246019370853901, -0.020241552963852882, 0.012013064697384834, 0.020676713436841965, 0.03966553136706352, -0.0007738932617940009, 0.014426227658987045, -0.0020307486411184072, -0.022245928645133972, 0.009072436019778252, 0.01978001929819584, 0.005630712024867535, -0.004677974618971348, 0.015494348481297493, -0.004598854575306177, 0.005940599367022514, -0.00819552130997181, -0.020083313807845116, 0.011314171366393566, 0.03628974407911301, 0.01606137491762638, 0.009474629536271095, -0.011320764198899269, 0.009876823052763939, -0.01603500172495842, 0.010206489823758602, -0.017129497602581978, -0.012909759767353535, -0.03502381965517998, 0.020742647349834442, 0.012909759767353535, -0.03805675730109215, -0.0194503515958786, 0.010503190569579601, -0.013311953283846378, -0.012982286512851715, -0.0009346058941446245, 0.02319536916911602, 0.016905322670936584, 0.013298766687512398, -0.00669883331283927, 0.03209637850522995, 0.018962444737553596, -0.0028994211461395025, -0.015903135761618614, 0.014689961448311806, -0.004559294320642948, -0.039138063788414, 0.0031400781590491533, 0.001860970165580511, -0.01913387142121792, 0.020821766927838326, 0.01727455109357834, 0.0035472167655825615, 0.001280756201595068, 0.002345580607652664, 0.008868042379617691, 0.0057856556959450245, -0.026900826022028923, 0.00485599460080266, 0.0030790898017585278, 0.003942816983908415, -0.0007755415863357484, -0.043990764766931534, 0.011986691504716873, 0.014953694306313992, -0.009679023176431656, -0.019318485632538795, -0.0013046570820733905, 0.00788563396781683, -0.013239426538348198, -0.000965924293268472, -0.01379326730966568, -0.002724697580561042, 0.005027421750128269, -0.022325048223137856, 0.008861448615789413, 0.0001372238912153989, 0.00853178184479475, 0.009777923114597797, 0.004038420505821705, -0.009850449860095978, -0.01544160209596157, 0.0025961275678128004, 0.007879041135311127, -0.01010758988559246, 0.019028378650546074, -0.022733835503458977, 0.027428293600678444, -0.018078938126564026, -0.005719722248613834, -0.0029950246680527925, 0.0010277368128299713, -0.017261363565921783, -0.004335120785981417, -0.025384359061717987, 0.0334150455892086, 0.012369105592370033, -0.013298766687512398, 0.0097845159471035, -0.0030065630562603474, -0.00020933854102622718, 0.005673568695783615, -0.023300861939787865, -0.03755566477775574, -0.02932717464864254, -0.006217519287019968, 0.023208556696772575, -0.0021494287066161633, -0.021177807822823524, 0.002104923827573657, -0.020465726032853127, 0.025305237621068954, 0.007048280443996191, 0.2367272675037384, 0.0013903705403208733, -0.0023587672039866447, 0.02522611804306507, -0.00507687171921134, 0.013384480029344559, 0.04095782712101936, 0.008505408652126789, -0.001080483547411859, -0.004028530791401863, -0.014083373360335827, 0.023749209940433502, -0.011446038261055946, 0.011149337515234947, -0.00674828328192234, -0.012408665381371975, -0.029195308685302734, -0.028641467913985252, 0.003903257194906473, 0.007035093382000923, 0.020558033138513565, -0.0028582129161804914, -0.03449635207653046, -0.029933761805295944, 0.02440854348242283, 0.0062999362125992775, 0.01466358732432127, 0.02980189584195614, 0.010733957402408123, 0.002294482197612524, 0.005874665919691324, 0.008755954913794994, 0.021138247102499008, 0.011709771119058132, 0.017670150846242905, -0.015560281462967396, 0.02405250445008278, -0.0171822439879179, 0.009098809212446213, 0.024593157693743706, 0.02829861454665661, 0.001102736103348434, -0.017867950722575188, -0.020848140120506287, 0.01379326730966568, 0.022733835503458977, -0.003603260265663266, -0.004476877860724926, -0.018725084140896797, 0.002660412574186921, -0.03760841116309166, -0.032755713909864426, 0.013872386887669563, 0.0484478622674942, -0.0017703117337077856, 0.004925224930047989, -0.00870980229228735, -0.01516468171030283, -0.026834892109036446, 0.0030659029725939035, -0.014083373360335827, 0.032491978257894516, 0.0003230736474506557, 0.04712919145822525, -0.014057000167667866, 0.023353610187768936, -0.037001822143793106, 0.0013302062870934606, -0.01178889162838459, -0.02245691604912281, 0.013872386887669563, -0.0007083719247020781, 0.0016483349027112126, 0.0047274245880544186, -0.021138247102499008, -0.013911946676671505, 0.0030279913917183876, 0.014267987571656704, 0.02610962651669979, 0.009250455535948277, 0.00682410690933466, 0.015402041375637054, 0.01217130571603775, 0.015929508954286575, -0.016456976532936096, -0.017498724162578583, -0.0018313000909984112, 0.004262594040483236, -0.017221802845597267, 0.0026159074623137712, 0.011914164759218693, -0.03652710095047951, -0.018369045108556747, 0.013964693993330002, -0.01289657223969698, -0.008953755721449852, 0.024157997220754623, 0.011742738075554371, -0.012606465257704258, -0.010707584209740162, -0.012560312636196613, 0.03032936342060566, 0.013272392563521862, 0.01217130571603775, -0.0033642516937106848, 0.0028961244970560074, -0.00336754834279418, -0.016667963936924934, 0.015322921797633171, 0.002790631027892232, -0.0049779717810451984, -0.0014117988757789135, 0.010450443252921104, 0.01516468171030283, 0.0032027147244662046, 0.008485628291964531, -0.010127370245754719, 0.0005060388357378542, -0.0004495833709370345, 0.004012047313153744, -0.0031878796871751547, -0.03230736404657364, -0.013199865818023682, -0.005301045253872871, -0.011815264821052551, -0.02257559634745121, -0.02042616717517376, -0.009824076667428017, -0.007259266916662455, -0.03407438099384308, -0.00341205345466733, -0.01530973520129919, 0.019964633509516716, -0.010918570682406425, -0.000311741343466565, 0.0008167499909177423, 0.004625227767974138, -0.0013335029361769557, 0.004931818228214979, 0.015322921797633171, -0.011630651541054249, -0.005370275117456913, 0.006590043194591999, -0.0009354300564154983, 0.027164559811353683, -0.03214912489056587, -0.015217428095638752, -0.015520721673965454, 0.0031878796871751547, -0.009593309834599495, 0.005680162459611893, -0.017525097355246544, 0.00046647878480143845, -0.020742647349834442, 0.026307426393032074, -0.012435038574039936, 0.004391164518892765, 0.006863666698336601, 0.013094373047351837, 0.0016079507768154144, 0.002788982819765806, 0.028667841106653214, 0.03283483162522316, -0.027744773775339127, -0.012250425294041634, 0.007246080320328474, -0.16836751997470856, 0.021995382383465767, -0.0009304850827902555, -0.028008507564663887, 0.04396438971161842, 0.02622830495238304, 0.005769172217696905, -0.011024064384400845, -0.013700960204005241, 0.006102135870605707, 0.008175740949809551, 0.011215271428227425, -0.021771207451820374, -0.014399854466319084, 0.003913147374987602, -0.009646056219935417, -0.02603050507605076, -0.007648274302482605, 0.021059127524495125, 0.015837201848626137, 0.011821858584880829, 0.010925164446234703, 0.003570293541997671, 0.0007083719247020781, 0.039190810173749924, -0.02319536916911602, -0.007226300425827503, 0.025687651708722115, -0.009441662579774857, -0.021230554208159447, -0.013081186451017857, 0.014966880902647972, 0.004368087742477655, 0.020795393735170364, 0.0024510740768164396, 0.012777892872691154, -0.005910929292440414, -0.036184247583150864, -0.000521698035299778, 0.02811400033533573, 0.016483349725604057, -0.0026521708350628614, 0.020623967051506042, -0.0016227858141064644, 0.015560281462967396, 0.007641681004315615, 0.004209847655147314, 0.0024889858905225992, 0.002723049372434616, -0.007714207749813795, 0.024949198588728905, -0.019054751843214035, 0.0012304820120334625, 0.008248267695307732, 0.0239865705370903, 0.01868552528321743, 0.0016302032163366675, 0.0033988666255027056, 0.0054526920430362225, -0.007312013767659664, -0.02935354970395565, -0.005891148932278156, 0.008670241571962833, -0.037265557795763016, -0.005739502143114805, -0.018316298723220825, -0.026953572407364845, 0.0025400840677320957, -0.026597533375024796, 0.011736145243048668, -0.007483440451323986, -0.023037130013108253, 0.005996642634272575, -0.013199865818023682, 0.013074592687189579, -0.001795036718249321, -0.03085683099925518, 0.02283932827413082, 0.009520783089101315, -0.00344831682741642, -0.02608325146138668, 0.02534479834139347, -0.009797702543437481, 0.0008612549863755703, 0.009355949237942696, 0.005930709186941385, 0.01178889162838459, -0.01317349262535572, -0.015665775164961815, -0.010720770806074142, 0.0026587643660604954, -0.013140526600182056, -0.01871189847588539, -0.022536035627126694, 0.027217306196689606, -0.004285670816898346, 0.01492732111364603, 0.019819580018520355, 0.004552701022475958, -0.013516346924006939, 0.008901008404791355, -0.0003255461633671075, -0.03457547351717949, 0.012454818934202194, 0.040509480983018875, -0.006059279199689627, 0.0021823954302817583, 0.02130967378616333, 0.024184370413422585, -0.02908981591463089, -0.010516377165913582, 0.016575656831264496, 0.00085466168820858, 0.03518206253647804, 0.021296488121151924, 0.012395478785037994, 0.02452722378075123, -0.03536667302250862, 0.014004253782331944, -0.012296578846871853, 0.024540411308407784, 0.019344858825206757, -0.004532921127974987, 0.024131624028086662, -0.03676446154713631, 0.010509783402085304, -0.09357267618179321, -0.01492732111364603, -0.004308747593313456, -0.00971198920160532, 0.023445915430784225, 0.021481100469827652, -0.0007611186592839658, -0.011755924671888351, 0.0035472167655825615, 0.00851859524846077, -0.022364608943462372, -0.010615277104079723, -0.011399884708225727, 0.004074683878570795, 0.032465606927871704, -0.028193121775984764, -0.015402041375637054, -0.02570083923637867, -0.012962506152689457, 0.02192944847047329, 0.002874696161597967, -0.011558124795556068, 0.009481222368776798, -0.014136120676994324, 0.005610932130366564, -0.023485476151108742, -0.04156441241502762, 0.025120625272393227, 0.006098839454352856, 0.001406029681675136, -0.0027032692451030016, -0.02233823575079441, 0.025410732254385948, -0.020241552963852882, -0.02252284809947014, 0.0031994180753827095, -0.025028318166732788, -0.028694214299321175, 0.038716092705726624, -0.006296639330685139, 0.02009649947285652, 0.022272301837801933, 0.005920819006860256, -0.02572721242904663, 0.006461473181843758, -0.005719722248613834, -0.017261363565921783, 0.016338296234607697, 0.00029958487721160054, -0.006013126112520695, -0.04881708696484566, 0.0004602975386660546, -0.03589414060115814, -0.008083434775471687, -0.001306305406615138, 0.01916024461388588, -0.013569093309342861, 0.03652710095047951, -0.0136614004150033, -0.016311923041939735, 0.02384151704609394, 0.026017319411039352, -0.016733895987272263, 0.00246426067315042, 0.00807684101164341, 0.0073186070658266544, 0.004625227767974138, -0.004964784719049931, 0.00870980229228735, 0.0007586461724713445, -0.002294482197612524, 0.024250304326415062, -0.03684358298778534, -0.009837263263761997, -0.03280846029520035, 0.01139329094439745, -0.030962323769927025, -0.002660412574186921, 0.008821888826787472, -0.012705366127192974, 0.007048280443996191, -0.02756015956401825, -0.007503220811486244, -0.0062603759579360485, 0.016786642372608185, 0.02856234833598137, 0.008122994564473629, 0.022417355328798294, 0.02042616717517376, -0.034760087728500366, 8.174710819730535e-05, 0.023010756820440292, 0.0029637061525136232, -0.01770971156656742, -0.017366856336593628, 0.03718643635511398, 0.01542841549962759, 0.022773396223783493, 0.016839390620589256, 0.011736145243048668, -0.026742586866021156, -0.0031170013826340437, -0.07141906023025513, 0.03233373910188675, -0.007780140731483698, 0.012296578846871853, 0.02505469135940075, -0.02195582166314125, 0.026267865672707558, -0.014821827411651611, 0.002060418715700507, -0.009659242816269398, -0.010793297551572323, 0.023300861939787865, -0.016549283638596535, -0.018725084140896797, -0.028667841106653214, -0.01492732111364603, 0.031990885734558105, 0.013977880589663982, 0.02078220620751381, -0.0017324000364169478, 0.009408695623278618, 0.0030543645843863487, 0.028034880757331848, 4.648819594876841e-05, 0.0025878858286887407, 0.012580092065036297, -0.0019845953211188316, 0.01632510870695114, -0.021111873909831047, -0.010555936954915524, 0.0010565826669335365, -0.02083495445549488, -0.00826804805546999, 0.004588964395225048, -0.012006471864879131, -0.03481283411383629, -0.007239487022161484, -0.006059279199689627, 0.01028561033308506, 0.0018477834528312087, 0.005775765515863895, -0.0023439323995262384, 0.013832827098667622, -0.00019182497635483742, -0.009817482903599739, -0.003304911544546485, -0.0016087748808786273, 0.03312493860721588, 0.02932717464864254, -0.00801750086247921, -0.00661641638725996, 0.017999818548560143, 0.012857012450695038, -0.013213053345680237, -0.006655976641923189, -0.024157997220754623, -0.006916413549333811, -0.02378877066075802, -0.005126321688294411, -0.03352053835988045, 0.03225461766123772, 0.009632869623601437, 0.012006471864879131, -0.03288757801055908, 0.018092123791575432, 0.015876762568950653, 0.0015477865235880017, -0.004684567917138338, 0.0012057570274919271, 0.0035538100637495518, -0.0026785442605614662, 0.008901008404791355, 0.008485628291964531, -0.016021816059947014, 0.02369646355509758, -0.010951537638902664, 0.019634965807199478, -0.022654715925455093, -0.0441490039229393, 0.01570533588528633, -0.01954265870153904, -0.014624027535319328, -0.03278208523988724, -0.005287858657538891, 0.005716425832360983, 0.023063503205776215, -0.00475379778072238, -0.012197678908705711, -0.028641467913985252, 0.02240416780114174, -0.009810890071094036, 0.012006471864879131, -0.0005138684064149857, 0.016997629776597023, 0.007298827171325684, 0.02080858126282692, -0.02063715271651745, -0.01830311119556427, 0.018883325159549713, 0.0170635636895895, 0.030487602576613426, -0.005983456037938595, 0.012131744995713234, -0.019859138876199722, -0.00325710978358984, -0.0013236129889264703, -0.012744925916194916, -0.032940324395895004, 0.010457037016749382, -0.002970299683511257, 0.025397544726729393, -0.001974705373868346, -0.011841638013720512, 0.003932927269488573, -0.017380043864250183, 0.0028730477206408978, -0.0336524061858654, -0.03781939670443535, -0.01727455109357834, 0.0015634456649422646, -0.008307607844471931, 0.007602120749652386, -0.017076749354600906, 0.008762548677623272, 0.019634965807199478, -0.00683729350566864, 0.016285549849271774, -0.014267987571656704, 0.011155931279063225, -0.012191085144877434, -0.014940507709980011, -0.013714146800339222, -0.005634008906781673, -0.02643929235637188, 0.01164383813738823, -0.009619683027267456, 0.005307638552039862, 0.044518228620290756, 0.004875774960964918, 0.05248298496007919, 0.009369135834276676, 0.006220816168934107, 0.004160397220402956, 0.008380134589970112, 0.016984444111585617, 0.004127430729568005, 0.005726315546780825, -0.026333799585700035, -0.017432790249586105, 0.01789432391524315, -0.010246049612760544, 0.0032323847990483046, -0.010865824297070503, -0.006046092603355646, 0.022325048223137856, 0.00835376139730215, 0.0012939429143443704, -0.029221681877970695, 0.015243801288306713, 0.027454666793346405, 0.0012568554375320673, 0.01691851019859314, 0.00029876071494072676, -0.008083434775471687, -0.004084574058651924, 0.009969130158424377, -0.010674617253243923, -0.010885603725910187, -0.006270266138017178, 0.014386667869985104, -0.009863636456429958, -0.008993315510451794, 0.011024064384400845, -0.02519974485039711, -0.03478645905852318, 0.001824706792831421, -0.012237238697707653, 0.017788831144571304, 0.016140496358275414, -0.02393382415175438, 0.02054484747350216, -0.004025233909487724, -0.026623906567692757, -0.003570293541997671, 0.002282943809404969, -0.014254800975322723, 0.003285131650045514, -0.00666916323825717]
    
    # Since this requires a call to the OpenAI API, I will not run it all the time
    # def test_get_ada_embedding(self):
    #     assert get_ada_embedding(self.question) == self.question_embedding
    
    def test_get_closest_nodes(self):
        summary_file = "./test/inputs/index.parquet"
        df_summary = pd.read_parquet(summary_file, engine="pyarrow")
        close = get_closest_nodes(df_summary, "embedding", self.question_embedding, threshold = 0.15)
        #q1 = 'For all requests to export gold in any form, they must be directed to the South African Diamond and Precious Metals Regulator.'
        assert close.iloc[0]['section_reference'] == 'C.(C)'
        #q2 = 'Acquiring gold for trade purposes requires approval from the South African Diamond and Precious Metals Regulator. Once approved, a permit must be obtained from SARS, which allows the holder to access gold allocation from Rand Refinery Limited.'
        assert close.iloc[1]['section_reference'] == 'C.(G)'
