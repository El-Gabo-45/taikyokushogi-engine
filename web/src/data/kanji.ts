/**
 * Kanji symbols for all Taikyoku Shogi piece types.
 *
 * Sources: English names and kanji from the Wikipedia "Game equipment" section
 * (https://en.wikipedia.org/wiki/Taikyoku_shogi).  The mapping key is the
 * two-letter (or three-letter) abbreviation used internally by the engine.
 */

const KANJI_MAP: Record<string, string> = {
  // ── Base pieces ──────────────────────────────────────────
  "K":   "王将",   // King
  "CP":  "太子",   // Crown Prince
  "GG":  "大将",   // Great General
  "G":   "金将",   // Gold General
  "RG":  "右将",   // Right General
  "LG":  "左将",   // Left General
  "RS":  "後旗",   // Rear Standard
  "Q":   "奔王",   // Free King
  "FT":  "奔獏",   // Free Dream-eater
  "WO":  "鳩槃",   // Wooden Dove
  "CD":  "鳩盤",   // Ceramic Dove
  "ED":  "地龍",   // Earth Dragon
  "FR":  "奔鬼",   // Free Demon
  "HR":  "走馬",   // Running Horse
  "BC":  "獣曹",   // Beast Cadet
  "LO":  "天狗",   // Tengu
  "MR":  "右山鷲", // Right Mountain Eagle
  "ML":  "左山鷲", // Left Mountain Eagle
  "DM":  "火鬼",   // Fire Demon
  "W":   "鯨鯢",   // Whale
  "RR":  "走兎",   // Running Rabbit
  "WT":  "白虎",   // White Tiger
  "TS":  "玄武",   // Turtle-snake
  "L":   "香車",   // Incense Chariot
  "RV":  "反車",   // Reverse Chariot
  "FG":  "香象",   // Fragrant Elephant
  "WE":  "白象",   // White Elephant
  "TD":  "山鳩",   // Turtle Dove
  "FS":  "飛燕",   // Flying Swallow
  "CO":  "禽吏",   // Fowl Officer
  "RA":  "雨龍",   // Rain Dragon
  "FO":  "森鬼",   // Forest Demon
  "MS":  "山鹿",   // Mountain Stag
  "RP":  "走狗",   // Running Pup
  "RU":  "走蛇",   // Running Serpent
  "SS":  "横蛇",   // Side Serpent
  "GR":  "大鳩",   // Great Dove
  "RT":  "走虎",   // Running Tiger
  "BA":  "走熊",   // Running Bear
  "YA":  "夜叉",   // Nature Spirit (Yaksha)
  "BD":  "羅刹",   // Buddhist Devil (Rakshasa)
  "GU":  "金剛",   // Guardian of the Gods (Kongo)
  "WR":  "力士",   // Sumo Wrestler (Rikishi)
  "S":   "銀将",   // Silver General
  "DE":  "酔象",   // Drunken Elephant
  "NK":  "近王",   // Neighboring King
  "GC":  "金車",   // Gold Chariot
  "SI":  "横龍",   // Side Dragon
  "RN":  "走鹿",   // Running Stag
  "RW":  "走狼",   // Running Wolf
  "BG":  "角将",   // Angle General
  "RO":  "飛将",   // Flying General
  "TT":  "右虎",   // Right Tiger
  "LT":  "左虎",   // Left Tiger
  "RI":  "右龍",   // Right Dragon
  "LE":  "左龍",   // Left Dragon
  "BO":  "獣吏",   // Beast Officer
  "WD":  "風龍",   // Wind Dragon
  "FP":  "奔狗",   // Free Pup
  "RB":  "行鳥",   // Rushing Bird
  "OK":  "古鵄",   // Old Kite
  "PC":  "孔雀",   // Peacock
  "WA":  "水龍",   // Water Dragon
  "FI":  "火龍",   // Fire Dragon
  "C":   "銅将",   // Copper General
  "PM":  "鳳師",   // Phoenix Master
  "KM":  "麟師",   // Kirin Master
  "SV":  "銀車",   // Silver Chariot
  "VE":  "竪熊",   // Vertical Bear
  "N":   "桂馬",   // Cassia Horse / Knight
  "PI":  "豚将",   // Pig General
  "CG":  "鶏将",   // Chicken General
  "PG":  "狗将",   // Pup General
  "H":   "馬将",   // Horse General
  "O":   "牛将",   // Ox General
  "CN":  "中旗",   // Center Standard
  "SA":  "横猪",   // Side Boar
  "SR":  "銀兎",   // Silver Rabbit
  "GL":  "金鹿",   // Gold Stag
  "LN":  "獅子",   // Lion
  "CT":  "禽曹",   // Fowl Cadet
  "GS":  "大鹿",   // Great Stag
  "VD":  "猛龍",   // Fierce Dragon
  "WL":  "林鬼",   // Woodland Demon
  "VG":  "副将",   // Vice General
  "CI":  "石車",   // Stone Chariot
  "CE":  "雲鷲",   // Cloud Eagle
  "B":   "角行",   // Angle Mover / Bishop
  "R":   "飛車",   // Flying Chariot / Rook
  "WF":  "横狼",   // Side Wolf
  "FC":  "飛猫",   // Flying Cat
  "MF":  "山鷹",   // Mountain Hawk
  "VT":  "竪虎",   // Vertical Tiger
  "SO":  "兵士",   // Soldier
  "LS":  "小旗",   // Little Standard
  "CL":  "雲龍",   // Cloud Dragon
  "CR":  "銅車",   // Copper Chariot
  "RH":  "走車",   // Running Chariot
  "HE":  "羊兵",   // Ram's-head Soldier
  "VO":  "猛牛",   // Fierce Ox
  "GD":  "大龍",   // Great Dragon
  "GO":  "金翅",   // Gold Bird
  "DS":  "無明",   // Dark Spirit (Avidya)
  "DV":  "提婆",   // Deva
  "WC":  "木車",   // Wood Chariot
  "WH":  "白駒",   // White Foal
  "DL":  "𠵇犬",   // Howling Dog (Left)
  "DR":  "𠵇犬",   // Howling Dog (Right)
  "SM":  "横行",   // Side Mover
  "PR":  "踊鹿",   // Prancing Stag
  "WB":  "水牛",   // Water Ox
  "FL":  "猛豹",   // Fierce Leopard
  "EG":  "猛鷲",   // Fierce Eagle
  "FD":  "飛龍",   // Flying Dragon
  "PS":  "毒蛇",   // Poisonous Serpent
  "FY":  "鳫飛",   // Flying Goose
  "ST":  "烏行",   // Strutting Crow
  "BI":  "盲犬",   // Blind Dog
  "WG":  "水将",   // Water General
  "F":   "火将",   // Fire General
  "PH":  "鳳凰",   // Phoenix
  "KR":  "麒麟",   // Kirin
  "HM":  "鉤行",   // Hook Mover
  "LL":  "小亀",   // Little Turtle
  "GT":  "大亀",   // Great Turtle
  "CA":  "摩羯",   // Capricorn
  "TC":  "瓦車",   // Tile Chariot
  "VW":  "竪狼",   // Vertical Wolf
  "SX":  "横牛",   // Side Ox
  "DO":  "驢馬",   // Donkey
  "FH":  "馬麟",   // Flying Horse
  "VB":  "猛熊",   // Fierce Bear
  "AB":  "嗔猪",   // Angry Boar
  "EW":  "悪狼",   // Evil Wolf
  "WI":  "風馬",   // Wind Horse
  "CK":  "鶏飛",   // Flying Chicken
  "OM":  "古猿",   // Old Monkey
  "CC":  "淮鶏",   // Huai Chicken
  "NB":  "北狄",   // Northern Barbarian
  "SU":  "南蛮",   // Southern Barbarian
  "WS":  "西戎",   // Western Barbarian
  "ES":  "東夷",   // Eastern Barbarian
  "VS":  "猛鹿",   // Fierce Stag
  "NT":  "猛狼",   // Fierce Wolf
  "TF":  "隠狐",   // Treacherous Fox
  "MT":  "中師",   // Center Master
  "PE":  "鵬師",   // Peng Master
  "EC":  "土車",   // Earth Chariot
  "VI":  "朱雀",   // Vermillion Sparrow
  "BL":  "青龍",   // Blue Dragon
  "EB":  "変狸",   // Enchanted Badger
  "HO":  "騎兵",   // Horseman
  "OW":  "鴟行",   // Swooping Owl
  "CM":  "登猿",   // Climbing Monkey
  "CS":  "猫刀",   // Cat Sword
  "SW":  "燕羽",   // Swallow's Wings
  "BM":  "盲猿",   // Blind Monkey
  "BT":  "盲虎",   // Blind Tiger
  "OC":  "牛車",   // Ox Chariot
  "SF":  "横飛",   // Side Flyer
  "BB":  "盲熊",   // Blind Bear
  "OR":  "老鼠",   // Old Rat
  "SQ":  "方行",   // Square Mover
  "SN":  "蟠蛇",   // Coiled Serpent
  "RD":  "臥龍",   // Reclining Dragon
  "FE":  "奔鷲",   // Free Eagle
  "LI":  "獅鷹",   // Lion Hawk
  "CH":  "車兵",   // Chariot Soldier
  "SL":  "横兵",   // Side Soldier
  "VR":  "竪兵",   // Vertical Soldier
  "WN":  "風将",   // Wind General
  "RE":  "川将",   // River General
  "M":   "山将",   // Mountain General
  "SD":  "前旗",   // Front Standard
  "HS":  "馬兵",   // Horse Soldier
  "GN":  "木将",   // Wood General
  "OS":  "牛兵",   // Ox Soldier
  "EA":  "土将",   // Earth General
  "BS":  "猪兵",   // Boar Soldier
  "SG":  "石将",   // Stone General
  "LP":  "豹兵",   // Leopard Soldier
  "T":   "瓦将",   // Tile General
  "BE":  "熊兵",   // Bear Soldier
  "I":   "鉄将",   // Iron General
  "GE":  "大旗",   // Great Standard
  "GM":  "大師",   // Great Master
  "RC":  "右車",   // Right Chariot
  "LC":  "左車",   // Left Chariot
  "MK":  "横猿",   // Side Monkey
  "VM":  "竪行",   // Vertical Mover
  "OX":  "飛牛",   // Flying Ox
  "LB":  "弩兵",   // Longbow Soldier
  "VP":  "竪狗",   // Vertical Pup
  "VH":  "竪馬",   // Vertical Horse
  "BN":  "炮兵",   // Cannon Soldier
  "DH":  "龍馬",   // Dragon Horse
  "DK":  "龍王",   // Dragon King
  "SE":  "刀兵",   // Sword Soldier
  "HF":  "角鷹",   // Horned Hawk
  "EL":  "飛鷲",   // Flying Eagle
  "SP":  "鎗兵",   // Spear Soldier
  "VL":  "竪豹",   // Vertical Leopard
  "TG":  "猛虎",   // Fierce Tiger
  "SC":  "弓兵",   // Crossbow Soldier
  "DG":  "吼犬",   // Roaring Dog
  "LD":  "狛犬",   // Lion Dog
  "D":   "犬",     // Dog
  "GB":  "仲人",   // Go-between
  "P":   "歩兵",   // Pawn
};

/** Look up the kanji abbreviation for a piece code. */
export function pieceKanji(abbrev: string): string {
  return KANJI_MAP[abbrev] ?? abbrev;
}