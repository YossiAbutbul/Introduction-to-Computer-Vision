// build_report.js - builds report.docx for Maman 11.
// Numbers in the tables and in the text are read from ../outputs/*.csv, so
// they always match the last run of the code.
// Run from maman01/report:  node build_report.js  [other_name.docx]

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, WidthType, BorderStyle, HeadingLevel, PageBreak, Footer, PageNumber, LevelFormat,
  Math: OMath, MathRun, MathSubScript, MathSuperScript,
} = require("docx");

const OUT = path.join(__dirname, "..", "outputs");
const FONT = "Calibri";
const SIZE = 22;                   // 11 pt

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
const t2 = readCsv(path.join(OUT, "task2", "results.csv"));
const t3 = readCsv(path.join(OUT, "task3", "summary.csv"));
const resAll = readCsv(path.join(OUT, "task1", "resolution_check.csv"));
const resGrid = resAll.find((x) => x.pair.startsWith("grid"));           // keypoints on the 1024 grid
const res = resAll.filter((x) => !x.pair.startsWith("grid"));

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
        // ". " ": " ", " ends a phrase - don't let a Latin block cross it
        // ("GPU. Kaggle: 4" would otherwise be shown as one reversed block)
        if (/[.,:;]/.test(text[j]) && text[j + 1] === " ") break;
        if (text[j] === " " && text[j + 1] === "(") break;        // "6 (0.795" -> "6" + " (0.795"
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
    spacing: { after: 100, line: 276 },
    ...opts,
    children: flat(content),
  });
}

// short bullet line
function li(content) {
  return new Paragraph({
    bidirectional: true,
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 276 },
    children: flat(content),
  });
}

// bold line that opens a step or a question
function label(text) {
  return para(textRuns(text, { bold: true, boldComplexScript: true }), { spacing: { before: 160, after: 60 }, keepNext: true });
}

const todo = (text) => textRuns(text, { highlight: "yellow" });

function heading(text, level) {
  return new Paragraph({ heading: level, bidirectional: true, children: textRuns(text) });
}

function caption(content) {
  return para(flat(content, { size: 18, sizeComplexScript: 18, italics: true }),
    { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 } });
}

// ---------------------------------------------------------------- math

const M = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [M(b)], subScript: [M(s)] });
const sup = (b, s) => new MathSuperScript({ children: [M(b)], superScript: [M(s)] });
const math = (...c) => new OMath({ children: c });

const eR = () => math(sub("e", "R"));
const et = () => math(sub("e", "t"));
const Nin = () => math(sub("N", "inliers"));
const rin = () => math(sub("r", "inliers"));
const FGT = () => math(sub("F", "GT"));
const e3d = () => math(sub("e", "3D"));
// "e_R=2.13°" inside one equation (text after an equation is placed wrong in RTL)
const eRv = (v) => math(sub("e", "R"), M("=" + v));
const etv = (v) => math(sub("e", "t"), M("=" + v));
const rinv = (v) => math(sub("r", "inliers"), M("=" + v));
const tHat = () => M("t̂");

// centred equation on its own line
function equation(...c) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 }, children: [math(...c)] });
}

// ---------------------------------------------------------------- tables

const line = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: line, bottom: line, left: line, right: line };

function cell(content, width, isHead) {
  const opts = { size: 19, sizeComplexScript: 19, bold: isHead, boldComplexScript: isHead };
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

// widths are relative, the table is centred
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
    children: [new ImageRun({ type: file.endsWith(".png") ? "png" : "jpg", data: fs.readFileSync(file),
      transformation: { width: widthPx, height: heightPx } })],
  });
}

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---------------------------------------------------------------- content

const body = [];
const add = (...x) => body.push(...x);

// ----- title page
add(
  para(textRuns("מבוא לראייה ממוחשבת (22928)", { size: 36, sizeComplexScript: 36, bold: true, boldComplexScript: true }),
    { alignment: AlignmentType.CENTER, spacing: { before: 3600, after: 200 } }),
  para(textRuns("ממ\"ן 11, סמסטר 2027א", { size: 28, sizeComplexScript: 28 }),
    { alignment: AlignmentType.CENTER, spacing: { after: 1600 } }),
  para(["שם: ", ...todo("[שם מלא]")], { alignment: AlignmentType.CENTER }),
  para(["ת.ז.: ", ...todo("[מספר]")], { alignment: AlignmentType.CENTER }),
  pageBreak(),
);

// ----- code and data
add(
  heading("הקוד והנתונים", HeadingLevel.HEADING_1),
  li("כל סקריפט מריצים מתוך התיקייה maman01. הפלטים (טבלאות ואיורים) נשמרים ב-outputs."),
  li("הפונקציות המשותפות נמצאות ב-utils, וכל הפרמטרים בקובץ utils/config.py."),
  table(["קובץ", "משימה"], [
    ["task1_matching_pose.py", "1"], ["task1_resolution_check.py", "1 (בדיקה נוספת)"], ["task2_viewpoint.py", "2"], ["task3_reconstruction.py", "3, 4"],
  ], [3.2, 1.2]),
  para(""),
  heading("הערות על הנתונים", HeadingLevel.HEADING_2),
  li("במאגר 11 תמונות בגודל 3072×2048, ולכל תמונה קובץ עם K, R ומרכז המצלמה C. מטריצת K זהה בכל התמונות."),
  li(["R בקובץ הוא סיבוב ממצלמה לעולם, לכן השתמשתי ב-", math(sup("R", "T")),
    " (בלי השחלוף חציון מרחק Sampson ב-0005-0006 יוצא 6810 במקום 0.027)."]),
  li(["R שמור עם 6 ספרות בלבד. בזוויות קטנות זה משנה את ", eR(), ", כי ", math(M("tr(R)≈3-"), sup("θ", "2")),
    " (ב-SIFT על 0005-0006 יצא 0.089° במקום 0.106°). לכן החלפתי כל R במטריצת הסיבוב הקרובה ביותר (SVD)."]),
  pageBreak(),
);

// ---------------------------------------------------------------- task 1

const pairs = pairSel.filter((r) => r.chosen === "1").sort((a, b) => Number(a.gap) - Number(b.gap));
const P = pairs.map((r) => r.pair);                     // small, medium, large
const r1 = (pair, m) => get(ratio, pair, m);
const m1 = (pair, m) => get(mnn, pair, m);
const spS = r1(P[0], "superpoint"), siftS = r1(P[0], "sift"), siftL = r1(P[2], "sift");
const orbS = r1(P[0], "orb"), orbL = r1(P[2], "orb");
const gpuSpeedup = meanTime(ratio, "superpoint") / meanTime(ratioKaggle, "superpoint");
const img1 = (pair, m) => path.join(OUT, "task1", "matches_" + pair + "_" + m + ".jpg");
const rs = (pair, m, side) => res.find((x) => x.pair === pair && x.method === m && Number(x.long_side) === side);
const avgT = (m, side) => { const t = res.filter((x) => x.method === m && Number(x.long_side) === side).map((x) => Number(x.time_s)); return t.reduce((a, b) => a + b, 0) / t.length; };
const resSlow = avgT("superpoint", 3072) / avgT("superpoint", 1024);

add(
  heading("משימה 1", HeadingLevel.HEADING_1),

  heading("שלב 1: בחירת הזוגות", HeadingLevel.HEADING_2),
  li("שלושה זוגות במרחק של 1, 3 ו-5 תמונות ברצף, כולם סביב אמצע הרצף."),
  li("המצלמות מסודרות על קשת סביב המזרקה, ולכן ה-baseline וזווית הסיבוב גדלים ביחד. בפועל אלה שלוש רמות קושי."),
  table(["זוג", "baseline", "זווית סיבוב (מעלות)"],
    pairs.map((p) => [p.pair, f(p.baseline, 2), f(p.rot_angle_deg, 1)]), [1.8, 1.5, 1.8]),
  caption("טבלה 1: הזוגות שנבחרו (ערכי אמת)"),
  para(["כאשר: ", math(M("baseline=|"), sub("C", "2"), M("-"), sub("C", "1"), M("|")), ", ו-",
    math(sub("C", "1"), M(","), sub("C", "2")), " הם מיקומי המצלמות בעולם שהגיעו מקובצי המאגר."]),

  heading("שלב 2: גילוי והתאמה", HeadingLevel.HEADING_2),
  li("SIFT: הגדרות ברירת מחדל, מרחק L2."),
  li("ORB: עד 5000 נקודות, מרחק Hamming. עם ברירת המחדל (500) יש הרבה פחות נקודות מ-SIFT."),
  li("SuperPoint: מודל מאומן מהחבילה של LightGlue, עד 4096 נקודות, מרחק L2. השתמשתי רק בגילוי והתיאור."),
  li("SuperPoint מקטין את התמונה לצלע ארוכה של 1024 ומחזיר את הנקודות בגודל המקורי."),
  li("ההתאמה זהה לכל השיטות: BFMatcher ו-ratio test של 0.75. להשוואה הרצתי גם MNN."),

  heading("שלב 3: איכות ההתאמות", HeadingLevel.HEADING_2),
  li(["התאמה עקבית אם ", math(sub("d", "S"), M("<"), sup("τ", "2")), " ביחס ל-", FGT(), ", עם ", math(M("τ=2")), " פיקסלים."]),
  table(["זוג", "שיטה", "התאמות", Nin(), rin()],
    ratio.map((x) => [x.pair, NAME[x.method], x.n_matches, x.n_inliers_FGT, f(x.r_inliers_FGT, 3)]),
    [1.6, 1.5, 1.2, 1.2, 1.2]),
  caption(["טבלה 2: איכות ההתאמות ביחס ל-", FGT(), " (ratio test)"]),
  image(img1(P[0], "sift"), 600, 200),
  caption(["איור 1: SIFT על " + P[0] + ". ירוק: עקבית עם ", FGT(), ", אדום: לא עקבית (150 התאמות אקראיות)"]),
  image(img1(P[1], "sift"), 600, 200),
  caption("איור 2: SIFT על " + P[1]),
  image(img1(P[2], "sift"), 600, 200),
  caption("איור 3: SIFT על " + P[2] + "."),

  heading("שלב 4: שיערוך התנוחה", HeadingLevel.HEADING_2),
  li("findEssentialMat עם RANSAC (prob=0.999, סף של 1px) ואחריו recoverPose, עם K של המאגר. אותו דבר לכל השיטות."),
  li("הרצתי את אותו קוד גם ב-Kaggle וקיבלתי ב-ORB שגיאות תנוחה שונות, למרות שמספר ההתאמות ומספר ההתאמות העקביות יצאו זהים. " +
    "ב-SIFT וב-SuperPoint התוצאות יצאו זהות לגמרי. בדקתי ומצאתי ש-RANSAC של OpenCV בוחר דגימות לפי המקום שלהן ברשימה עם seed קבוע, " +
    "כך ששינוי הסדר של אותן התאמות לבד משנה את התוצאה. כנראה ש-ORB החזיר את ההתאמות בסדר אחר ב-Kaggle."),
  li(["הפיזור גדול. למשל ב-SuperPoint על " + P[0] + " ", et(), " נע בין " + f(spS.e_t_min, 2) + "° ל-" +
    f(spS.e_t_max, 2) + "° רק בגלל הסדר."]),
  li("לכן הרצתי RANSAC  30 פעמים על סדרים אקראיים של אותן התאמות (עם seed קבוע), ובטבלה מופיע החציון."),
  table(["זוג", "שיטה", "RANSAC inliers", [eR(), " (מעלות)"], [et(), " (מעלות)"]],
    ratio.map((x) => [x.pair, NAME[x.method], x.n_inliers_ransac, f(x.e_R_deg, 3), f(x.e_t_deg, 3)]),
    [1.6, 1.5, 1.5, 1.2, 1.2]),
  caption("טבלה 3: שגיאות התנוחה (חציון של 30 הרצות RANSAC)"),

  heading("שלב 5: זמן ריצה", HeadingLevel.HEADING_2),
  li("הזמן כולל רק גילוי, תיאור והתאמה של שתי התמונות. ריצת חימום אחת, ואז חציון של 5 ריצות."),
  li("בלפטופ שלי יש 16 threads ואין GPU. ב-Kaggle יש 4 ליבות ו-GPU מסוג Tesla T4, ורק SuperPoint משתמש ב-GPU."),
  li("בלפטופ הזמנים השתנו בין ריצות (SIFT על 0005-0006 בין 0.94 ל-2.75 שניות), אבל הסדר בין השיטות נשאר אותו דבר."),
  table(["שיטה", "לפטופ, CPU (שניות)", "Kaggle, GPU (שניות)"],
    ["sift", "orb", "superpoint"].map((m) => [NAME[m], f(meanTime(ratio, m), 2), f(meanTime(ratioKaggle, m), 2)]),
    [1.5, 1.8, 1.8]),
  caption("טבלה 4: זמן גילוי, תיאור והתאמה, ממוצע על שלושת הזוגות"),
);

add(
  heading("מסקנות", HeadingLevel.HEADING_3),
  li("מספר ההתאמות יורד מהר כשהזווית גדלה (ב-SIFT: " + P.map((p) => r1(p, "sift").n_matches).join(", ") +
    "), למרות שמספר הנקודות בכל תמונה כמעט לא משתנה. מה שנפגע זה ההתאמה."),
  li(["ל-SIFT יש את ה-", rin(), " הכי גבוה בכל הזוגות."]),
  li(["ב-SuperPoint ", rin(), " נמוך יותר וחציון Sampson גבוה בהרבה (" + f(spS.median_sampson_px2, 3) + " מול " +
    f(siftS.median_sampson_px2, 3) + "). בדקתי למה עם task1_resolution_check.py:"]),
  li("כמעט כל ההבדל הוא בהתאמות שמפספסות את הסף בקצת, 2 עד 6 פיקסלים מהקו האפיפולרי: " +
    f(rs(P[0], "superpoint", 1024).share_near_miss_2_6px * 100, 0) + "% ב-SuperPoint לעומת " +
    f(rs(P[0], "sift", 3072).share_near_miss_2_6px * 100, 0) + "% ב-SIFT. שגיאות גסות יש בשתיהן בערך באותה כמות."),
  li("הנקודות של SuperPoint יושבות בדיוק על הפיקסלים של התמונה המוקטנת (" + f(Number(resGrid.share_near_miss_2_6px.split(" ").pop()) * 100, 0) +
    "% מקואורדינטות ה-x שלמות), כלומר בצעדים של 3 פיקסלים בתמונה המקורית, בלי דיוק בתוך הפיקסל."),
  li("כשמריצים את SuperPoint ברזולוציה מלאה הוא כמעט משתווה ל-SIFT (טבלה 5). SIFT על תמונה מוקטנת כמעט לא נפגע בזוג הקטן ובבינוני, " +
    "כי הוא מחשב מיקום בתוך הפיקסל."),
  li("כלומר הבעיה היא ההקטנה ביחד עם חוסר הדיוק בתוך הפיקסל. ברזולוציה מלאה SuperPoint איטי פי " + f(resSlow, 0) +
    " בערך על המעבד, ו-SIFT עדיין יותר מדויק בתנוחה."),
  table(["שיטה ורזולוציה (צלע ארוכה)", ...P.map((p) => [rin(), " " + p])],
    [["superpoint", 1024], ["superpoint", 2048], ["superpoint", 3072], ["sift", 3072], ["sift", 1024]].map(([m, side]) =>
      [NAME[m] + ", " + side + (m === "superpoint" && side === 1024 ? " (ברירת מחדל)" : m === "sift" && side === 3072 ? " (רגיל)" : ""),
       ...P.map((p) => f(rs(p, m, side).r_inliers_FGT, 3))]),
    [2.6, 1.3, 1.3, 1.3]),
  caption("טבלה 5: שיעור ההתאמות העקביות לפי הרזולוציה שבה רצה כל שיטה"),
  li("SIFT הכי מדויק בתנוחה בכל הזוגות. בזוג הקטן ORB קצת יותר טוב מ-SuperPoint, בבינוני הם שקולים, ובגדול SuperPoint הרבה יותר טוב."),
  li(["בזוג הגדול ORB לא אמין: חציון ", eRv(f(orbL.e_R_deg, 2) + "°"), ", ובחלק מההרצות עד " + f(orbL.e_R_max, 0) + "°."]),
  li("את הסדר בין השיטות בדקתי גם עם 200 הרצות ושלושה seeds שונים, והוא לא השתנה."),
  li([rin(), " גבוה לא מבטיח תנוחה טובה: ב-ORB על הזוג הקטן ", rinv(f(orbS.r_inliers_FGT, 3)), " אבל ", et(),
    " גדול פי " + f(Number(orbS.e_t_deg) / Number(siftS.e_t_deg), 1) + " מ-SIFT."]),
  li(["עם RANSAC אחד אפשר בקלות להגיע למסקנה הפוכה. בהרצה אחת SuperPoint יצא אצלי הכי מדויק בזוג הבינוני ובגדול."]),
  li(["MNN נותן יותר התאמות אבל יותר שגויות. בזוג הגדול ", rin(), " של SIFT יורד מ-" + f(siftL.r_inliers_FGT, 3) + " ל-" +
    f(m1(P[2], "sift").r_inliers_FGT, 3) + ", ו-ORB עם MNN נכשל (", eRv(f(m1(P[2], "orb").e_R_deg, 0) + "°"), "). לכן ratio test."]),
  li("ORB הכי מהיר ו-SuperPoint הכי איטי על CPU. כש-SuperPoint רץ על GPU הוא מהיר פי 10 עד " + f(gpuSpeedup, 0) + " מאשר על הלפטופ."),
  li("למשימה 2 בחרתי ב-SIFT, כי הוא הכי טוב גם באיכות ההתאמות וגם בתנוחה."),
  pageBreak(),
);

// ---------------------------------------------------------------- task 2

const im = (k) => t2.find((x) => Number(x.image) === k);
const T2 = path.join(OUT, "task2");

add(
  heading("משימה 2", HeadingLevel.HEADING_1),
  heading("שלב 1: מה הרצתי", HeadingLevel.HEADING_2),
  li("SIFT. תמונה 0000 מול כל אחת מהתמונות 0001 עד 0010, לפי הסדר."),
  li("אותם פרמטרים כמו במשימה 1, כולל 30 הרצות של RANSAC."),

  heading("שלב 2: תוצאות", HeadingLevel.HEADING_2),
  table(["תמונה", "baseline", "זווית (מעלות)", "התאמות", Nin(), rin(), "RANSAC inliers", [eR(), " (מעלות)"], [et(), " (מעלות)"]],
    t2.map((x) => [x.image, f(x.baseline, 2), f(x.rot_angle_deg, 1), x.n_matches, x.n_inliers_FGT,
      f(x.r_inliers_FGT, 3), x.n_inliers_ransac, f(x.e_R_deg, 3), f(x.e_t_deg, 3)]),
    [0.8, 1, 1, 0.9, 0.9, 0.9, 1.1, 1, 1]),
  caption("טבלה 6: תמונה 0000 מול כל אחת מהתמונות"),
  image(path.join(T2, "plot_inliers.png"), 600, 208),
  caption("איור 4: מספר ההתאמות העקביות ושיעורן"),
  image(path.join(T2, "plot_errors.png"), 400, 277),
  caption("איור 5: שגיאות התנוחה (סקאלה לוגריתמית, הפס הבהיר הוא הטווח של 30 ההרצות)"),
  image(path.join(T2, "plot_geometry.png"), 600, 208),
  caption("איור 6: ה-baseline וזווית הסיבוב האמיתיים"),

  label("1. מספר ההתאמות ושיעור ההתאמות העקביות"),
  li([Nin(), " יורד מהר כבר מההתחלה: " + im(1).n_inliers_FGT + " בתמונה 1, " + im(2).n_inliers_FGT +
    " בתמונה 2, ו-" + im(10).n_inliers_FGT + " בתמונה 10."]),
  li([rin(), " כמעט יציב עד תמונה 6 (" + f(im(6).r_inliers_FGT, 3) + " בזווית של " + f(im(6).rot_angle_deg, 0) +
    "°), ואחרי זה נופל: " + [7, 8, 9, 10].map((k) => f(im(k).r_inliers_FGT, 3)).join(", ") + "."]),
  li("בהתחלה פשוט יש פחות התאמות ורובן נכונות. מבערך 60° גם החלק השגוי מתחיל לגדול."),

  label("2. דיוק התנוחה לאורך הרצף"),
  li(["עד תמונה 7 התנוחה מדויקת: ", eR(), " בין " + f(im(1).e_R_deg, 3) + "° ל-" + f(im(7).e_R_deg, 3) + "°, ו-", et(), " עד 0.2°."]),
  li(["בתמונה 8 זה עדיין עובד (", eRv(f(im(8).e_R_deg, 2) + "°"), ") עם " + im(8).n_inliers_ransac + " inliers בלבד."]),
  li(["בתמונות 9 ו-10 השיערוך נכשל, ", eR(), " מעל 100°. אין ירידה הדרגתית, זה עובד ואז פתאום לא."]),

  label("3. הקשר בין איכות ההתאמות לשגיאות"),
  li("הקשר לא לינארי. הירידה מ-" + im(1).n_inliers_FGT + " ל-" + im(7).n_inliers_FGT + " התאמות עקביות כמעט לא שינתה את השגיאה."),
  li(["הכישלון מגיע כש-", rin(), " יורד מתחת לבערך 0.25 ונשארות מעט התאמות נכונות."]),
  li(["בתמונה 9, RANSAC מצא " + im(9).n_inliers_ransac + " inliers, יותר מ-" + im(9).n_inliers_FGT +
    " ההתאמות שבאמת עקביות עם ", FGT(), ". כלומר הוא בחר מודל שגוי שקבוצה של התאמות שגויות תומכת בו."]),

  label("4. השפעת ה-baseline וזווית הסיבוב"),
  li("הזווית פוגעת בהתאמה: מזווית אחרת כל נקודה נראית אחרת, אז יש פחות התאמות ויותר מהן שגויות."),
  li(["ה-baseline עוזר לכיוון ההזזה: ", et(), " יורד מ-" + f(im(1).e_t_deg, 3) + "° בתמונה 1 ל-" + f(im(7).e_t_deg, 3) +
    "° בתמונה 7, למרות שיש פי " + f(im(1).n_matches / im(7).n_matches, 0) + " פחות התאמות."]),
  li([eR(), " לעומת זאת עולה לאט."]),
  li("במאגר הזה שניהם גדלים ביחד, אז אי אפשר להפריד ביניהם לגמרי."),

  label("5. איפה השגיאות הגדולות ולמה"),
  li("בתמונות 9 ו-10 (" + f(im(9).rot_angle_deg, 0) + "° ו-" + f(im(10).rot_angle_deg, 0) + "°)."),
  li("החלק המשותף לשתי התמונות הוא בעיקר המזרקה, והיא נראית בזווית חדה מאוד (איור 7)."),
  li("הרבה התאמות שגויות הולכות לחלונות של הארמון, שבכלל לא מופיע בתמונה 0000."),
  li("המרקם חוזר על עצמו (אבני הקיר, החלונות), ואני חושב שזה עוזר להתאמות שגויות לעבור את ה-ratio test. את זה לא בדקתי."),
  image(path.join(T2, "matches_0000-0010.jpg"), 600, 200),
  caption("איור 7: SIFT על 0000-0010. ירוק: עקבית, אדום: לא עקבית"),
  pageBreak(),
);

// ---------------------------------------------------------------- task 3

const pr = (name) => t3.find((x) => x.pair === name);
const A = pr("0005-0006"), B = pr("0004-0007"), C = pr("0003-0008");     // A = main pair
const T3 = path.join(OUT, "task3");
const removed = t3.reduce((s, x) => s + Number(x.removed_non_finite) + Number(x.removed_w) +
  Number(x.removed_neg_depth) + Number(x.removed_small_angle), 0);

add(
  heading("משימה 3", HeadingLevel.HEADING_1),

  heading("שלב 1: ההתאמות", HeadingLevel.HEADING_2),
  li("הזוג " + A.pair + " עם SIFT."),
  li(["מתוך " + A.n_matches + " התאמות השארתי רק את " + A.n_consistent + " שעקביות עם ", FGT(), ". אותן התאמות נכנסו לשני השחזורים."]),

  heading("שלב 2: התנוחה המשוערכת", HeadingLevel.HEADING_2),
  li(["את ", math(M("R̂")), " ו-", math(tHat()), " שיערכתי מכל ההתאמות, כמו במשימה 1."]),
  li("מתוך 30 ההרצות של RANSAC לקחתי את זו עם הכי הרבה inliers, כי זה מה שאפשר לעשות גם בלי ground truth."),
  li(["לתנוחה הזאת ", eRv(f(A.e_R_deg, 3) + "°"), " ו-", etv(f(A.e_t_deg, 3) + "°"), "."]),

  heading("שלב 3: מטריצות ההטלה וטריאנגולציה", HeadingLevel.HEADING_2),
  li("המצלמה הראשונה היא הראשית. הטריאנגולציה עם cv2.triangulatePoints."),
  equation(sub("P", "1"), M("="), sub("K", "1"), M("[I | 0]")),
  equation(sub("P", "2,GT"), M("="), sub("K", "2"), M("["), sub("R", "GT"), M(" | "), sub("t", "GT"), M("]")),
  equation(sub("P", "2,est"), M("="), sub("K", "2"), M("[R̂ | s"), tHat(), M("],    s=‖"), sub("t", "GT"), M("‖ / ‖"), tHat(), M("‖")),

  heading("שלב 4: סינון", HeadingLevel.HEADING_2),
  li("אותם קריטריונים בשני השחזורים. נקודה נשארת רק אם היא עוברת בשניהם, כך שהאינדקסים תואמים."),
  li("קואורדינטות סופיות, והרכיב הרביעי |w| גדול מ-10⁻¹⁰."),
  li("עומק חיובי בשתי המצלמות."),
  li("זווית של לפחות 2° בין שתי קרני הראייה (פחות מזה העומק לא אמין)."),
  li(removed === 0 ? "בכל שלושת הזוגות אף נקודה לא נפסלה." : "בסך הכל נפסלו " + removed + " נקודות."),

  heading("שלב 5: תוצאות", HeadingLevel.HEADING_2),
  li("את אותו תהליך הרצתי גם על שני הזוגות האחרים ממשימה 1, בשביל ההשוואה במשימה 4."),
  table(["זוג", "baseline", "נקודות", [e3d(), " ממוצע"], [e3d(), " חציון"], "ביחס לעומק", "reprojection עם GT (פיקסלים)", "reprojection משוערך (פיקסלים)"],
    [A, B, C].map((x) => [x.pair, f(x.baseline, 2), x.n_points_kept, f(x.e3d_mean, 4), f(x.e3d_median, 4),
      f(x.e3d_rel_median_pct, 2) + "%", f(x.reproj_gt_median_px, 3), f(x.reproj_est_median_px, 3)]),
    [1.3, 0.9, 0.8, 1, 1, 1, 1.2, 1.3]),
  caption("טבלה 7: השחזור בשלושת הזוגות (חציונים). הזוג הראשי הוא " + A.pair),
  image(path.join(T3, "cloud_" + A.pair + ".png"), 600, 206),
  caption("איור 8: שני ענני הנקודות על אותם צירים. כחול: תנוחה אמיתית, אדום: משוערכת"),
  image(path.join(T3, "e3d_vs_depth_" + A.pair + ".png"), 400, 277),
  caption(["איור 9: ", e3d(), " של כל נקודה מול העומק שלה"]),
  li("באיור 8 כמעט לא רואים הבדל, כי השגיאה היא בערך " + f(A.e3d_rel_median_pct, 1) + "% מהעומק. באיור 9 רואים את ההבדלים."),
  pageBreak(),
);

// ---------------------------------------------------------------- task 4

const factor = Number(A.e3d_rel_median_pct) / Number(B.e3d_rel_median_pct);
const etFactor = Number(A.e_t_deg) / Number(B.e_t_deg);

add(
  heading("משימה 4", HeadingLevel.HEADING_1),

  label("1. ההבדלים בין שני העננים"),
  li("כמעט כל השגיאה (" + f(Number(A.e3d_along_ray_median) * 100, 0) + "% בחציון) היא לאורך קרן הראייה, כלומר בעומק."),
  li("הענן המשוערך קרוב יותר למצלמה בערך ב-" + f((1 - Number(A.depth_ratio_est_gt_median)) * 100, 1) + "%."),
  li("חוץ מזה צורת הענן כמעט זהה."),

  label("2. איך שגיאות בתנוחה משפיעות על הנקודות"),
  li("השגיאה שיטתית: הממוצע של וקטורי ההפרש הוא " + f(Number(A.e3d_systematic) * 100, 0) +
    "% מהאורך הממוצע שלהם, אז כמעט כל הנקודות זזות לאותו כיוון."),
  li("זה הגיוני, כי לכל הנקודות יש אותה טעות בתנוחה."),
  li("השגיאה גדלה עם העומק (איור 9): בערך 0.015 בעומק 5, בערך 0.06 בעומק 9, ו-" + f(A.e3d_max, 2) + " בנקודה הרחוקה (עומק 25)."),

  label("3. הקשר בין שגיאת ההטלה לאיכות השחזור"),
  li("בזוג הראשי שגיאת ההטלה של השחזור המשוערך אפילו קטנה יותר (" + f(A.reproj_est_median_px, 3) + " מול " +
    f(A.reproj_gt_median_px, 3) + " פיקסלים), למרות שהנקודות זזו."),
  li("הטריאנגולציה מוצאת את הנקודה שמתאימה הכי טוב למטריצות ההטלה שקיבלה, גם אם הן קצת שגויות."),
  li("התנוחה המשוערכת חושבה מאותן התאמות רועשות, אז היא מתאימה להן אפילו יותר מהאמיתית."),
  li("כלומר שגיאת הטלה קטנה אומרת שהשחזור עקבי עם התמונות, לא שהוא נכון."),

  label("4. למה baseline קטן גורם לשגיאות גדולות בעומק"),
  li("כשה-baseline קטן שתי הקרניים כמעט מקבילות (בזוג הראשי " + f(A.tri_angle_median_deg, 1) +
    "° ביניהן). הזזה קטנה של קרן אחת מזיזה את נקודת החיתוך הרבה לאורך הקרן."),
  li("בזוג הראשי השגיאה היחסית היא " + f(A.e3d_rel_median_pct, 2) + "%, וב-" + B.pair + " רק " + f(B.e3d_rel_median_pct, 2) + "%, פי " + f(factor, 0) + " פחות."),
  li("ב-" + B.pair + " ה-baseline הוא " + f(B.baseline, 2) + " והזווית בין הקרניים " + f(B.tri_angle_median_deg, 0) + "°."),
  li(["חלק מזה כי שם גם ", et(), " קטן יותר, אבל רק פי " + f(etFactor, 1) + ". זה לא מסביר הבדל של פי " + f(factor, 0) + "."]),
  li(["ב-" + C.pair + " ה-baseline גדול עוד יותר, אבל התנוחה פחות מדויקת (", eRv(f(C.e_R_deg, 2) + "°"), "), ולכן השגיאה עולה ל-" + f(C.e3d_rel_median_pct, 2) + "%."]),

  label("5. האם הרבה נקודות זה שחזור טוב"),
  li("לא. בזוג הראשי יש הכי הרבה נקודות (" + A.n_points_kept + ") והשחזור הכי פחות מדויק ביחס לעומק."),
  li("ב-" + B.pair + " יש " + B.n_points_kept + " נקודות והשחזור הכי מדויק."),
  li(["נקודה יכולה להיות עקבית עם ", FGT(), " ועדיין לבוא מהתאמה שגויה, אם שתי הנקודות נמצאות במקרה על אותו קו אפיפולרי."]),
  li("מה שקובע זה דיוק התנוחה, ה-baseline ודיוק המיקום של הנקודות, לא כמה נקודות יש."),
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
      headingStyle("Heading1", "Heading 1", 32, 240, 0),
      headingStyle("Heading2", "Heading 2", 26, 200, 1),
      headingStyle("Heading3", "Heading 3", 24, 160, 2),
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, process.argv[2] || "report.docx");   // optional other file name
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
