// build_report.js - builds report.docx for Maman 11.
// Numbers in the tables are read from ../outputs/*.csv, so they always match
// the last run of the code.
// Run from maman01/report:  node build_report.js

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, WidthType, BorderStyle, HeadingLevel, PageBreak, Footer, PageNumber,
  Math: OMath, MathRun, MathSubScript, MathSuperScript,
} = require("docx");

const OUT = path.join(__dirname, "..", "outputs");
const FONT = "David";
const SIZE = 24;                   // 12 pt
const CONTENT_W = 9638;            // A4 width minus 2 cm margins, in DXA

// ---------------------------------------------------------------- data

function readCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => {
    const v = l.split(",");
    const r = {};
    head.forEach((h, i) => (r[h] = v[i]));
    return r;
  });
}

const ratio = readCsv(path.join(OUT, "task1", "results_ratio.csv"));
const mnn = readCsv(path.join(OUT, "task1", "results_mnn.csv"));
const pairSel = readCsv(path.join(OUT, "task1", "pair_selection.csv"));
const ratioKaggle = readCsv(path.join(OUT, "task1_kaggle", "results_ratio.csv"));   // timing only

const NAME = { sift: "SIFT", orb: "ORB", superpoint: "SuperPoint" };
const f = (x, d) => Number(x).toFixed(d);
const get = (rows, pair, method) => rows.find((r) => r.pair === pair && r.method === method);
const meanTime = (rows, method) => {
  const t = rows.filter((r) => r.method === method).map((r) => Number(r.time_median_s));
  return t.reduce((a, b) => a + b, 0) / t.length;
};

// ---------------------------------------------------------------- text helpers

// Word reverses "0.027 px²" if the whole run is right-to-left, so every string
// is split into Hebrew (RTL) parts and Latin / number (LTR) parts.
const HEB = /[֐-׿]/;
const LTR_CHAR = /[A-Za-z0-9²³⁰-₟Ͱ-Ͽ%°_]/;

function splitDirections(text) {
  const parts = [];
  let i = 0;
  while (i < text.length) {
    if (LTR_CHAR.test(text[i])) {
      let j = i, last = i;
      while (j < text.length && !HEB.test(text[j])) {
        if (LTR_CHAR.test(text[j])) last = j;
        j++;
      }
      parts.push({ s: text.slice(i, last + 1), rtl: false });
      i = last + 1;
    } else {
      let j = i;
      while (j < text.length && !LTR_CHAR.test(text[j])) j++;
      parts.push({ s: text.slice(i, j), rtl: true });
      i = j;
    }
  }
  return parts;
}

const textRuns = (text, opts = {}) => splitDirections(text).map((p) =>
  new TextRun({ text: p.s, rightToLeft: p.rtl, font: FONT, size: SIZE, sizeComplexScript: SIZE, ...opts }));

// strings -> runs, everything else (math, runs) is kept as is
const flat = (content, opts) => (Array.isArray(content) ? content : [content])
  .flatMap((c) => (typeof c === "string" ? textRuns(c, opts) : [c]));

function para(content, opts = {}) {
  return new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.BOTH,
    spacing: { after: 120, line: 320 },
    ...opts,
    children: flat(content),
  });
}

const todo = (text) => textRuns(text, { highlight: "yellow" });

function heading(text, level) {
  const size = level === HeadingLevel.HEADING_1 ? 30 : 26;
  return new Paragraph({
    heading: level,
    bidirectional: true,
    children: textRuns(text, { bold: true, boldComplexScript: true, size, sizeComplexScript: size }),
  });
}

function caption(content) {
  return para(flat(content, { size: 20, sizeComplexScript: 20 }),
    { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 } });
}

// ---------------------------------------------------------------- inline math

const M = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [M(b)], subScript: [M(s)] });
const sup = (b, s) => new MathSuperScript({ children: [M(b)], superScript: [M(s)] });
const math = (...c) => new OMath({ children: c });

const eR = () => math(sub("e", "R"));
const et = () => math(sub("e", "t"));
const Nin = () => math(sub("N", "inliers"));
const rin = () => math(sub("r", "inliers"));
const FGT = () => math(sub("F", "GT"));
const dS = () => math(sub("d", "S"));

// ---------------------------------------------------------------- tables

const line = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: line, bottom: line, left: line, right: line };

function cell(content, width, isHead) {
  const opts = { size: 20, sizeComplexScript: 20, bold: isHead, boldComplexScript: isHead };
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 80, right: 80 },
    children: [new Paragraph({
      bidirectional: true,
      alignment: AlignmentType.CENTER,
      children: flat(typeof content === "number" ? String(content) : content, opts),
    })],
  });
}

// widths are relative, the table is centred and as wide as it needs to be
function table(header, rows, widths) {
  const w = widths.map((x) => Math.round(x * 1000));
  const total = w.reduce((a, b) => a + b, 0);
  return new Table({
    visuallyRightToLeft: true,
    alignment: AlignmentType.CENTER,
    width: { size: total, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, w[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, w[i], false)) })),
    ],
  });
}

function image(file, widthPx, heightPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120 },
    keepNext: true,
    children: [new ImageRun({ type: "jpg", data: fs.readFileSync(file),
      transformation: { width: widthPx, height: heightPx } })],
  });
}

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---------------------------------------------------------------- content

const body = [];
const add = (...x) => body.push(...x);

// title page
add(
  para(textRuns("מבוא לראייה ממוחשבת (22928)", { size: 36, sizeComplexScript: 36, bold: true, boldComplexScript: true }),
    { alignment: AlignmentType.CENTER, spacing: { before: 3600, after: 200 } }),
  para(textRuns("ממ\"ן 11, סמסטר 2027א", { size: 28, sizeComplexScript: 28 }),
    { alignment: AlignmentType.CENTER, spacing: { after: 1600 } }),
  para(["שם: ", ...todo("[שם מלא]")], { alignment: AlignmentType.CENTER }),
  para(["ת.ז.: ", ...todo("[מספר]")], { alignment: AlignmentType.CENTER }),
  pageBreak(),
);

// code
add(
  heading("הקוד", HeadingLevel.HEADING_1),
  para("את כל הסקריפטים מריצים מתוך התיקייה maman01, והפלטים נשמרים בתיקייה outputs. " +
    "הפונקציות המשותפות נמצאות בתיקייה utils."),
  table(
    ["קובץ", "משימה"],
    [
      ["task1_matching_pose.py", "1"],
      ["task2_viewpoint.py", "2"],
      ["task3_reconstruction.py", "3, 4"],
    ],
    [3.2, 1.2],
  ),
  para(""),
);

// ---------------------------------------------------------------- task 1

const pairs = pairSel.filter((r) => r.chosen === "1").sort((a, b) => Number(a.gap) - Number(b.gap));
const P = pairs.map((r) => r.pair);                     // small, medium, large
const r = (pair, m) => get(ratio, pair, m);
const spSmall = r(P[0], "superpoint");

add(
  heading("1. משימה 1", HeadingLevel.HEADING_1),
  heading("בחירת הזוגות", HeadingLevel.HEADING_2),
  para("בחרתי שלושה זוגות שבהם המרחק ברצף בין התמונות הוא 1, 3 ו-5, כולם סביב אמצע הרצף. " +
    "המצלמות במאגר מסודרות על קשת סביב המזרקה, ולכן כשה-baseline גדל גם זווית הסיבוב גדלה, " +
    "ואי אפשר לבחור אותם בנפרד. בפועל אלה שלוש רמות קושי."),
  table(
    ["זוג", "baseline", "זווית סיבוב [°]"],
    pairs.map((p) => [p.pair, f(p.baseline, 2), f(p.rot_angle_deg, 1)]),
    [1.8, 1.5, 1.8],
  ),
  caption("טבלה 1: הזוגות שבחרתי (ערכי אמת)"),

  heading("איך עשיתי את זה", HeadingLevel.HEADING_2),
  para("בשלוש השיטות ההתאמה ושיערוך התנוחה זהים, ורק הגילוי והמתארים שונים. ב-SIFT השתמשתי בהגדרות ברירת המחדל. " +
    "ב-ORB הגדרתי 5000 נקודות, כי עם ברירת המחדל (500) הוא מוצא הרבה פחות נקודות מ-SIFT וההשוואה לא הוגנת. " +
    "ב-SuperPoint השתמשתי במודל המאומן מהחבילה של LightGlue עם עד 4096 נקודות, בלי ה-matcher של LightGlue. " +
    "SuperPoint מקטין את התמונה כך שהצלע הארוכה היא 1024, ומחזיר את הנקודות בגודל המקורי."),
  para(["את ההתאמה עשיתי עם BFMatcher (מרחק L2, וב-ORB מרחק Hamming) ו-ratio test של 0.75. להשוואה הרצתי גם MNN. " +
    "התאמה נחשבת עקבית אם ", math(sub("d", "S"), M("<"), sup("τ", "2")), " עם ", math(M("τ=2")),
    " פיקסלים ביחס ל-", FGT(), ". את התנוחה שיערכתי עם findEssentialMat (RANSAC עם prob=0.999 וסף של פיקסל) " +
    "ו-recoverPose, עם K של המאגר. הזמן כולל רק גילוי, תיאור והתאמה של שתי התמונות. " +
    "לפני המדידה יש ריצת חימום אחת, ואז אני לוקח חציון של 5 ריצות."]),
  para(["בקבצי המצלמות של המאגר R הוא סיבוב ממצלמה לעולם, ולכן השתמשתי ב-", math(sup("R", "T")),
    ". בלי השחלוף חציון מרחק Sampson של ההתאמות ב-0005-0006 יצא 6810 במקום 0.027. " +
    "בנוסף, R שמור בקובץ עם 6 ספרות בלבד, ובזוויות קטנות זה משנה את ", eR(), " כי ",
    math(M("tr"), M("(R)≈3−"), sup("θ", "2")), ". ב-SIFT על 0005-0006 קיבלתי 0.089° במקום 0.106°. " +
    "לכן החלפתי כל R במטריצת הסיבוב הכי קרובה אליה (עם SVD)."]),

  heading("RANSAC והסדר של ההתאמות", HeadingLevel.HEADING_2),
  para(["כשהרצתי את אותו קוד ב-Kaggle, ב-ORB יצאה תנוחה אחרת למרות שההתאמות היו אותן התאמות. " +
    "בדקתי וראיתי ש-RANSAC של OpenCV בוחר את הדגימות לפי המקום שלהן ברשימה ועם seed קבוע, " +
    "כך שאם ההתאמות מגיעות בסדר אחר התוצאה משתנה. ההבדל לא קטן: ב-SuperPoint על 0005-0006 ", et(),
    " נע בין " + f(spSmall.e_t_min, 2) + "° ל-" + f(spSmall.e_t_max, 2) + "° רק בגלל הסדר. " +
    "לכן הרצתי את RANSAC 30 פעם על סדרים אקראיים של אותן התאמות (עם seed קבוע שלי, כך שהתוצאה זהה בכל מחשב), " +
    "ובטבלאות מופיע החציון."]),
);

// results
add(
  heading("תוצאות", HeadingLevel.HEADING_2),
  table(
    ["זוג", "שיטה", "התאמות", Nin(), rin()],
    ratio.map((x) => [x.pair, NAME[x.method], x.n_matches, x.n_inliers_FGT, f(x.r_inliers_FGT, 3)]),
    [1.6, 1.5, 1.2, 1.2, 1.2],
  ),
  caption(["טבלה 2: איכות ההתאמות ביחס ל-", FGT()]),
  table(
    ["זוג", "שיטה", "RANSAC inliers", [eR(), " [°]"], [et(), " [°]"]],
    ratio.map((x) => [x.pair, NAME[x.method], x.n_inliers_ransac, f(x.e_R_deg, 3), f(x.e_t_deg, 3)]),
    [1.6, 1.5, 1.5, 1.2, 1.2],
  ),
  caption("טבלה 3: שגיאות התנוחה (חציון של 30 הרצות RANSAC)"),
  table(
    ["שיטה", "לפטופ, CPU [s]", "Kaggle, GPU T4 [s]"],
    ["sift", "orb", "superpoint"].map((m) => [NAME[m], f(meanTime(ratio, m), 2), f(meanTime(ratioKaggle, m), 2)]),
    [1.5, 1.8, 1.8],
  ),
  caption("טבלה 4: זמן גילוי, תיאור והתאמה, ממוצע על שלושת הזוגות"),
  para("בלפטופ יש 16 threads ואין GPU. ב-Kaggle יש 4 ליבות ו-Tesla T4, ורק SuperPoint משתמש ב-GPU. " +
    "בלפטופ הזמנים השתנו לא מעט בין ריצות, כנראה לפי העומס על המחשב. למשל SIFT על 0005-0006 לקח בריצות שונות " +
    "בין 0.94 ל-2.75 שניות, ו-SuperPoint בין 4.3 ל-6 שניות. הסדר בין השיטות היה זהה בכל הריצות."),
);

const img = (pair, m) => path.join(OUT, "task1", "matches_" + pair + "_" + m + ".jpg");
add(
  image(img(P[0], "sift"), 600, 200),
  caption(["איור 1: SIFT על " + P[0] + ". ירוק: עקבית עם ", FGT(), ", אדום: לא עקבית (150 התאמות אקראיות)"]),
  image(img(P[2], "sift"), 600, 200),
  caption("איור 2: SIFT על " + P[2]),
  para("את כל 9 התמונות (3 זוגות כפול 3 שיטות) אפשר למצוא ב-outputs/task1."),
);

// discussion (numbers come from the CSVs)
const m = (pair, method) => get(mnn, pair, method);
const siftS = r(P[0], "sift"), siftL = r(P[2], "sift"), orbS = r(P[0], "orb"), orbL = r(P[2], "orb");
const gpuSpeedup = meanTime(ratio, "superpoint") / meanTime(ratioKaggle, "superpoint");

add(
  heading("דיון", HeadingLevel.HEADING_2),
  para("מספר ההתאמות יורד מהר כשהזווית בין המצלמות גדלה. ב-SIFT יש " + siftS.n_matches + " התאמות בזוג הקטן, " +
    r(P[1], "sift").n_matches + " בבינוני ו-" + siftL.n_matches + " בגדול, למרות שמספר הנקודות שהוא מוצא בכל תמונה כמעט לא משתנה. " +
    "אז מה שנפגע זה ההתאמה עצמה. מזווית אחרת אותה נקודה נראית אחרת, המתארים כבר לא דומים מספיק וה-ratio test זורק אותם."),
  para(["ל-SIFT יש את ה-", rin(), " הכי גבוה בכל הזוגות (" + P.map((p) => f(r(p, "sift").r_inliers_FGT, 3)).join(", ") + "). " +
    "ב-SuperPoint הוא נמוך יותר, וגם חציון מרחק Sampson שלו גבוה בהרבה (" + f(spSmall.median_sampson_px2, 3) +
    " לעומת " + f(siftS.median_sampson_px2, 3) + " בזוג הקטן). אני חושב שזה בגלל ש-SuperPoint עובד על תמונה מוקטנת בערך פי 3, " +
    "ולכן המיקום של הנקודות פחות מדויק וחלק מההתאמות הנכונות לא עוברות את הסף של 2 פיקסלים. את זה לא בדקתי."]),
  para(["SIFT נתן את התנוחה הכי מדויקת בכל שלושת הזוגות, גם ב-", eR(), " וגם ב-", et(), ". " +
    "בזוג הקטן ORB קצת יותר טוב מ-SuperPoint, בבינוני הם בערך אותו דבר, ובגדול SuperPoint הרבה יותר טוב. " +
    "בזוג הגדול ORB כבר לא אמין: בחציון ", eR(), "=" + f(orbL.e_R_deg, 2) + "°, ובחלק מההרצות התנוחה יצאה שגויה לגמרי (עד " +
    f(orbL.e_R_max, 0) + "°). את הסדר בין השיטות בדקתי גם עם 200 הרצות של RANSAC ושלושה seeds שונים, והוא לא השתנה."]),
  para(["מה שהפתיע אותי הוא ש-", rin(), " גבוה לא אומר שהתנוחה טובה. ב-ORB על הזוג הקטן ", rin(), "=" + f(orbS.r_inliers_FGT, 3) +
    " ובכל זאת ", et(), "=" + f(orbS.e_t_deg, 2) + "°, יותר מפי 2 מ-SIFT."]),
  para(["הפיזור בין הרצות של RANSAC גדול. ב-SIFT על הזוג הקטן ", et(), " נע בין " + f(siftS.e_t_min, 2) + "° ל-" +
    f(siftS.e_t_max, 2) + "°, וזה יותר מההבדל בין השיטות. עם הרצה אחת אפשר בקלות להגיע למסקנה הפוכה, " +
    "וזה בדיוק מה שקרה לי בהתחלה: בהרצה אחת SuperPoint יצא הכי מדויק בזוג הבינוני ובגדול."]),
  para(["עם MNN יש יותר התאמות, אבל הרבה יותר מהן שגויות, ובזוג הגדול זה כבר מפריע. ה-", rin(), " של SIFT יורד שם מ-" +
    f(siftL.r_inliers_FGT, 3) + " ל-" + f(m(P[2], "sift").r_inliers_FGT, 3) + ", ", et(), " בחציון עולה מ-" + f(siftL.e_t_deg, 2) +
    "° ל-" + f(m(P[2], "sift").e_t_deg, 2) + "°, ו-ORB עם MNN לא מצליח בכלל (", eR(), "=" + f(m(P[2], "orb").e_R_deg, 0) + "°). " +
    "ה-ratio test זורק התאמות שאין להן התאמה אחת ברורה, וזה בדיוק מה שקורה כשהזווית גדלה, ולכן השתמשתי בו."]),
  para("ORB הכי מהיר ו-SuperPoint הכי איטי על המעבד. ב-Kaggle, כש-SuperPoint רץ על ה-GPU, הוא מהיר פי 10 עד " +
    f(gpuSpeedup, 0) + " מאשר על הלפטופ (תלוי באיזו ריצה של הלפטופ משווים). SIFT ו-ORB דווקא יותר איטיים שם, כי יש רק 4 ליבות."),
  para("למשימה 2 בחרתי ב-SIFT, כי הוא הכי טוב גם באיכות ההתאמות וגם בתנוחה, בכל אחד מהזוגות."),
  pageBreak(),
);

// remaining tasks
add(
  heading("2. משימה 2", HeadingLevel.HEADING_1), para(todo("[טרם נכתב]")),
  heading("3. משימה 3", HeadingLevel.HEADING_1), para(todo("[טרם נכתב]")),
  heading("4. משימה 4", HeadingLevel.HEADING_1), para(todo("[טרם נכתב]")),
);

// ---------------------------------------------------------------- document

const headingStyle = (id, name, size, before, outline) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { font: FONT, size, sizeComplexScript: size, bold: true, boldComplexScript: true, color: "000000" },
  paragraph: { spacing: { before, after: 120 }, keepNext: true, outlineLevel: outline },
});

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SIZE, sizeComplexScript: SIZE } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 30, 360, 0),
      headingStyle("Heading2", "Heading 2", 26, 240, 1),
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, process.argv[2] || "report.docx");   // optional other file name
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
